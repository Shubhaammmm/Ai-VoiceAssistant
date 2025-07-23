from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings

from qdrant_client.http.models import Distance, VectorParams
from api.config import Config
from dotenv import load_dotenv
import os

load_dotenv()


embeddings = OllamaEmbeddings(model="mxbai-embed-large")

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="maharashtra_begging_act",
    embeddings=embeddings,
)

def retrieve_context(query: str, k: int = 3) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join(doc.page_content for doc in docs)
