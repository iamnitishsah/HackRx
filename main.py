import os
import time
import asyncio
import hashlib
import requests
import tempfile
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configuration values
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "hackrx"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-1.5-pro"
NAMESPACE = "policy"
TOP_K = 5

embedder = None
pinecone_index = None
llm = None
processed_document_hashes = set()

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class HackrxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackrxResponse(BaseModel):
    answers: List[str]

class PDFProcessor:
    @staticmethod
    def get_url_hash(url: str) -> str:
        # Hash the PDF URL for caching
        return hashlib.md5(url.encode()).hexdigest()

    @staticmethod
    def download_pdf(url: str) -> bytes:
        # Download PDF file
        response = requests.get(str(url), timeout=60)
        response.raise_for_status()
        return response.content

    @staticmethod
    def split_pdf(pdf_bytes: bytes) -> List[Any]:
        # Save PDF to temp file and split it into chunks
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            temp_path = tmp_file.name

        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                add_start_index=True
            )
            chunks = splitter.split_documents(docs)
        finally:
            os.remove(temp_path)
        return chunks

class PineconeHandler:
    @staticmethod
    def initialize_index():
        # Initialize Pinecone index
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        if not pc.has_index(INDEX_NAME):
            pc.create_index(
                name=INDEX_NAME,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        return pc.Index(INDEX_NAME)

    @staticmethod
    def store_chunks(chunks: List[Any], index, embedder, document_hash: str):
        # Check if chunks already stored, then embed and store if not
        try:
            existing = index.query(
                vector=[0.0] * 3072,
                top_k=1,
                namespace=NAMESPACE,
                filter={"doc_hash": document_hash},
                include_metadata=True
            )
            if existing.matches:
                print(f"Document {document_hash} already exists in index. Skipping...")
                return
        except Exception:
            pass

        for chunk in chunks:
            if hasattr(chunk, 'metadata'):
                chunk.metadata["doc_hash"] = document_hash

        vector_store = PineconeVectorStore(
            index=index,
            embedding=embedder,
            namespace=NAMESPACE,
            text_key="page_content"
        )
        vector_store.add_documents(chunks)

class PolicyQAEngine:
    @staticmethod
    async def answer_questions(questions: List[str], embedder, index, llm_model) -> List[str]:
        # Embed all questions
        question_embeddings = embedder.embed_documents(questions)

        # Retrieve documents concurrently for each question
        async def retrieve_documents(question: str, embedding: List[float]) -> List[Document]:
            loop = asyncio.get_event_loop()
            def _query():
                results = index.query(
                    vector=embedding,
                    top_k=TOP_K,
                    namespace=NAMESPACE,
                    include_metadata=True
                )
                documents = []
                for match in results.matches:
                    metadata = match.metadata if match.metadata else {}
                    if "page_content" in metadata:
                        documents.append(Document(
                            page_content=metadata["page_content"],
                            metadata={**metadata, "similarity_score": match.score}
                        ))
                return documents
            return await loop.run_in_executor(None, _query)

        retrieval_tasks = [
            retrieve_documents(q, emb)
            for q, emb in zip(questions, question_embeddings)
        ]
        relevant_docs = await asyncio.gather(*retrieval_tasks)

        # Prepare prompts for LLM
        prompts = [
            PolicyQAEngine.prepare_prompt(q, docs) if docs else f"No relevant information found for: {q}"
            for q, docs in zip(questions, relevant_docs)
        ]

        # Call LLM for answers (rate limited)
        semaphore = asyncio.Semaphore(5)
        async def llm_call(prompt: str) -> str:
            async with semaphore:
                await asyncio.sleep(0.05)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    response = await loop.run_in_executor(executor, lambda: llm_model.invoke(prompt))
                return response.strip()

        answers = await asyncio.gather(*[llm_call(p) for p in prompts])
        return answers

    @staticmethod
    def prepare_prompt(question: str, docs: List[Document]) -> str:
        # Build LLM prompt using document chunks
        prompt = f"""You are an insurance policy expert. Answer this question using ONLY the information provided below.

Question: {question}

Policy Information:
"""
        for i, doc in enumerate(docs, 1):
            page_info = f" (Page {doc.metadata.get('page', 'N/A')})" if doc.metadata.get('page') else ""
            prompt += f"\n{i}.{page_info} {doc.page_content.strip()}\n"
        prompt += """
Instructions:
- Give a direct, clear answer
- Use exact information from the policy clauses above
- If information is not available, say "Not specified in the policy"
- Keep the answer concise but complete

Answer:"""
        return prompt

async def load_components():
    """Initialize global components."""
    global embedder, pinecone_index, llm
    embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
    )
    pinecone_index = PineconeHandler.initialize_index()
    llm = GoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load all required components
    await load_components()
    yield
    # Shutdown: Print shutdown message only
    print("Shutting down...")

app = FastAPI(
    title="Super Optimized Insurance Policy Assistant API",
    description="Maximum performance API for insurance policy Q&A",
    version="3.0.0",
    lifespan=lifespan
)

def ingest_pdf_document(pdf_url: str) -> bool:
    # Download, chunk, and embed a new PDF if not already processed
    try:
        document_hash = PDFProcessor.get_url_hash(pdf_url)
        if document_hash in processed_document_hashes:
            print(f"Document already processed: {document_hash[:8]}...")
            return True
        pdf_bytes = PDFProcessor.download_pdf(pdf_url)
        chunks = PDFProcessor.split_pdf(pdf_bytes)
        PineconeHandler.store_chunks(chunks, pinecone_index, embedder, document_hash)
        processed_document_hashes.add(document_hash)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )

@app.post("/hackrx/run", response_model=HackrxResponse)
async def super_optimized_policy_qa(request: HackrxRequest):
    """
    API endpoint for uploading a policy PDF and asking questions.
    """
    try:
        ingest_pdf_document(str(request.documents))
        answers = await PolicyQAEngine.answer_questions(
            request.questions, embedder, pinecone_index, llm
        )
        return HackrxResponse(answers=answers)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )