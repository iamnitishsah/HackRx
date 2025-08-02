import os
import sys
import time
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "hackrx"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-1.5-pro"
NAMESPACE = "policy"
TOP_K = 5


class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


def init_embedding():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )


def init_pinecone_index():
    client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    return client.Index(INDEX_NAME)


def retrieve_relevant_chunks(query, embedding, index, k=TOP_K, namespace=NAMESPACE):
    print(f"Retrieving top {k} relevant document chunks for the query...")
    query_embedding = embedding.embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=k,
        namespace=namespace,
        include_metadata=True
    )

    documents = []
    for match in results.matches:
        metadata = match.metadata if match.metadata else {}
        if "page_content" in metadata:
            documents.append(
                Document(
                    page_content=metadata["page_content"],
                    metadata={**metadata, "similarity_score": match.score}
                )
            )
    print(f"Found {len(documents)} relevant chunks.")
    return documents


def build_prompt(query, documents):
    prompt_intro = (
        "You are an insurance policy assistant.\n"
        "Answer the user query by analyzing the relevant policy document clauses provided below.\n"
        "Provide a clear, concise answer based on the policy information.\n"
        "Use the exact information from the clauses to answer the question.\n"
        "User Query: "
    )
    prompt_query = f"\"{query}\"\n\nRelevant document clauses:\n"

    clause_texts = ""
    for i, doc in enumerate(documents, 1):
        page = doc.metadata.get("page", "Unknown")
        start_idx = doc.metadata.get("start_index", "Unknown")
        clause_texts += f"Clause {i} (Page: {page}, Index: {start_idx}):\n{doc.page_content.strip()}\n\n"

    prompt = prompt_intro + prompt_query + clause_texts + "\n\nAnswer:"
    return prompt


def call_gemini_llm(prompt):
    llm = GoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.0,
        # max_tokens=1000,
        google_api_key=GOOGLE_API_KEY,
    )
    response = llm.invoke(prompt)
    return response


def main():
    user_query = input("Enter your insurance query: ").strip()
    start_time = time.time()
    embedding = init_embedding()
    index = init_pinecone_index()
    docs = retrieve_relevant_chunks(user_query, embedding, index, TOP_K)

    if not docs:
        print("⚠️ No relevant document chunks found for your query.")
        return

    prompt = build_prompt(user_query, docs)

    llm_response = call_gemini_llm(prompt)

    print(f"\nAnswer: {llm_response.strip()}")

    print(f"\n⏱️ Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()