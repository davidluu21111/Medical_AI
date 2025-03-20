from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# --- Your other imports ---
from src.helper import download_hugging_face_embed
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt
from groq_llm import OfficialGroqLLM

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# --- Setup your embeddings, vector store, etc. ---
index_name = "medicalbot"
embeddings = download_hugging_face_embed()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- Initialize your Groq LLM ---
groq_llm = OfficialGroqLLM(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.4,
    max_tokens=500
)

# --- Create RAG chain ---
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(groq_llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Using form data => request.form["msg"]
    user_message = request.form["msg"]
    print("User said:", user_message)

    response = rag_chain.invoke({"input": user_message})
    bot_reply = response["answer"]
    print("Bot reply:", bot_reply)

    # Return plain text
    return bot_reply

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
