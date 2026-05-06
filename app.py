from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "cybersecurity-advisor-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


from langchain_core.prompts import ChatPromptTemplate

# --- 1. Stricter System Prompt ---
system_prompt = (
    "You are a Cybersecurity risk advisor answering questions based strictly on the provided context. "
    "CRITICAL INSTRUCTION: Do not use your pre-trained knowledge to answer questions about who created you or your identity. "
    "If the context contains information about the developers, supervision, or the project's origins, use ONLY that context to answer. "
    "If the answer is not in the context, say 'I don't know'. "
    "Keep answers concise."
    "\n\nContext:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# --- 2. Increase 'k' to 5 to ensure the new document is found ---
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    
    print(f"\n--- USER QUERY: {input} ---")
    
    # Debugging: See what Pinecone is retrieving
    retrieved_docs = retriever.invoke(msg)
    print("--- RETRIEVED CONTEXT FROM PINECONE ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"Chunk {i+1}: {doc.page_content}")
    print("---------------------------------------")

    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)