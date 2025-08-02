import os
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "hackrx"
EMBEDDING_MODEL = "models/gemini-embedding-001"
NAMESPACE = "policy"
TOP_K = 5


class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


def init_embedding():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)


def init_pinecone_index():
    client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    return client.Index(INDEX_NAME)


def search(query, embedding, index, top_k=TOP_K, namespace=NAMESPACE):
    print(f"Searching for: '{query}'")
    vector = embedding.embed_query(query)
    result = index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    docs = []
    for match in result.matches:
        md = match.metadata or {}
        content = md.get("page_content", "")
        docs.append(Document(content, {**md, "score": match.score}))
    print(f"Found {len(docs)} documents")
    return docs


def display_documents(docs):
    if not docs:
        print("No relevant documents found.")
        return
    print(f"Top {len(docs)} relevant chunks:\n")
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "Unknown")
        score = doc.metadata.get("score", 0)
        print(f"--- Chunk #{i} (Page: {page}, Score: {score:.3f}) ---")
        print(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
        print("-" * 50)


def main():
    query = input("Enter your question: ").strip()
    if not query:
        print("Empty query. Exiting.")
        return

    start_time = time.time()
    embedding = init_embedding()
    index = init_pinecone_index()
    docs = search(query, embedding, index)
    display_documents(docs)
    print(f"\nTotal query time: {time.time() - start_time:.2f} seconds")

    if docs:
        print("\nThese chunks are ready for answer generation with Gemini 2.5 Pro.")


if __name__ == "__main__":
    main()