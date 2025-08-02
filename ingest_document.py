import os
import time
import requests
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Constants
INDEX_NAME = "hackrx"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def download_pdf(url):
    start = time.time()
    print(f"Downloading PDF from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    elapsed = time.time() - start
    print(f"✅ Downloaded PDF in {elapsed:.2f} seconds")
    return response.content

# def split_pdf_chunks(pdf_bytes):
#     start = time.time()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(pdf_bytes)
#         tmp_path = tmp_file.name
#
#     try:
#         loader = UnstructuredPDFLoader(
#             tmp_path,
#             mode="single",
#             unstructured_kwargs={"strategy": "fast", "no_ocr": True}
#         )
#         docs = loader.load()
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#             add_start_index=True
#         )
#         chunks = splitter.split_documents(docs)
#     finally:
#         os.remove(tmp_path)
#
#     elapsed = time.time() - start
#     print(f"✅ Split PDF into {len(chunks)} chunks in {elapsed:.2f} seconds")
#     return chunks


def split_pdf_chunks(pdf_bytes):
    start = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
        )
        chunks = splitter.split_documents(docs)
    finally:
        os.remove(tmp_path)


    elapsed = time.time() - start
    print(f"✅ Split PDF into {len(chunks)} chunks in {elapsed:.2f} seconds")
    return chunks

def init_pinecone():
    start = time.time()
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if not pc.has_index(INDEX_NAME):
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    elapsed = time.time() - start
    print(f"✅ Initialized Pinecone index '{INDEX_NAME}' in {elapsed:.2f} seconds")
    return index

def embed_and_store(chunks, index):
    start = time.time()
    print("Embedding chunks and uploading to Pinecone...")
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embedder,
        namespace="policy",
        text_key="page_content"
    )
    vector_store.add_documents(chunks)
    elapsed = time.time() - start
    print(f"✅ Embedded and stored {len(chunks)} chunks in Pinecone in {elapsed:.2f} seconds")

def main():
    PDF_URL = input("PDF URL: ")
    overall_start = time.time()
    pdf_bytes = download_pdf(PDF_URL)
    chunks = split_pdf_chunks(pdf_bytes)
    index = init_pinecone()
    embed_and_store(chunks, index)
    elapsed = time.time() - overall_start
    print(f"🚀 Full ingestion pipeline completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()