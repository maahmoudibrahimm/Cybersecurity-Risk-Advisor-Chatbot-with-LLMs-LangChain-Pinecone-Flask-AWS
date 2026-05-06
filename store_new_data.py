import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings

# 1. Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("Loading embedding model...")
# 2. Initialize Embeddings (Must match the model used to create the index)
embeddings = download_hugging_face_embeddings()

# 3. Connect to Existing Pinecone Index
index_name = "cybersecurity-advisor-chatbot"
print(f"Connecting to existing Pinecone index: {index_name}...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 4. Create the new Document
info = Document(
    page_content="This project was implemented as an application of the Expert Systems course under the supervision of Dr. Reham El-Anany and Eng. Youssef. It was developed by students from the Computer and Control Engineering Department at the Faculty of Engineering.",
    metadata={"source": "Expert Systems"}
)

# 5. Add the Document to the existing index
print("Adding new document to Pinecone...")
docsearch.add_documents(documents=[info])
print("Document added successfully!")

# --- The Eventual Consistency Fix ---
print("Waiting 10 seconds for Pinecone to index the new data...")
time.sleep(10) 

# 6. Test the Retrieval
print("\n--- Testing Retrieval ---")
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

query = "Who are the developers of this project?"
print(f"Querying: '{query}'")

retrieved_docs = retriever.invoke(query)

# 7. Print the results clearly
for i, doc in enumerate(retrieved_docs):
    print(f"\nResult {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")