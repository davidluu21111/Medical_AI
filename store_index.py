from src.helper import load_pdf, text_split,download_hugging_face_embed
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

data_loaded = load_pdf(data="DATA/")
text_chunks= text_split(data_loaded)
embeddings = download_hugging_face_embed()

pc = Pinecone(api_key = PINECONE_API_KEY)
index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

docsearch = PineconeVectorStore.from_documents (
    documents = text_chunks,
    index_name = index_name,
    embedding= embeddings
)


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
