# Cybersecurity Risk Advisor Chatbot — Project Documentation

**Institution:** Faculty of Engineering, Suez Canal University

**Course:** Expert Systems

**Authors / Team:** Mahmoud Ibrahim, Marwan Mohamed, Mohamed Islam, Abdel Hamid Nael, Ahmed Kandil, Abdel Rahman Hany, Fadi Ashraf

**Supervisors:** Dr. Reham El-Anany, Eng. Youssef

---

**Abstract**

This document describes the design, implementation, and usage of the "Cybersecurity Risk Advisor" — an expert-systems-style chatbot that answers cybersecurity questions using a knowledge base built from domain-specific books and documents. The project uses Retrieval-Augmented Generation (RAG) to ground answers in a curated knowledge base stored as vector embeddings in Pinecone. The application stack includes Python, LangChain, Pinecone, HuggingFace sentence-transformers embeddings, and a Flask web front-end. This documentation covers concept, architecture, RAG background, code walkthrough for each script, setup and testing instructions, limitations, and extensions.

---

**Table of Contents**

1. Introduction and Project Concept
2. Objectives and Use Cases
3. Background: Expert Systems & Chatbots
4. Background: Retrieval-Augmented Generation (RAG)
   - 4.1 Core Concepts
   - 4.2 Operational Flow
   - 4.3 RAG in this Project
5. System Architecture and Components
6. Data pipeline: PDF ingestion to vector index
7. File-by-file code explanations
   - 7.1 `app.py`
   - 7.2 `src/helper.py`
   - 7.3 `src/prompt.py`
   - 7.4 `store_index.py`
   - 7.5 `store_new_data.py`
   - 7.6 `check_models.py`
   - 7.7 `templates/chat.html` and `static/style.css`
   - 7.8 `requirements.txt`, `README.md`, `Dockerfile` and `setup.py`
8. Deployment and running instructions
9. Evaluation, testing, and validation
10. Limitations, risks and mitigation
11. Future work and enhancements
12. Conclusion
13. References and Appendix

---

**1. Introduction and Project Concept**

This project implements an expert-systems-style question-answering chatbot specialized in cybersecurity. Its goal is to provide concise, context-grounded responses about security risks, vulnerabilities, policies, and best practices. The system constructs a knowledge base from academic and reference books (PDFs) and converts content into dense vector embeddings. At query-time, the system retrieves relevant text chunks from the index and conditions a modern generative model to produce short, accurate answers referencing that retrieved content.

Key design choices:

- Use RAG (Retrieval-Augmented Generation) to reduce hallucinations and ground answers.
- Use sentence-transformers (`all-MiniLM-L6-v2`) for embeddings (384 dimensions) for a compact and efficient vector space.
- Use Pinecone as a managed vector store for scalable similarity search.
- Use LangChain and a generative model (via `ChatGroq` in the current code) to compose retrieved context and produce the final answer.
- Provide a simple Flask front-end (`templates/chat.html`) for interacting with the assistant.

**2. Objectives and Use Cases**

- Educational use in the "Expert Systems" course: demonstrate RAG and LLM-based knowledge systems.
- Provide students and staff with a quick cybersecurity advisor that references curated textbooks and notes.
- Serve as a foundation for more advanced expert systems: policy assistants, compliance checkers, or vulnerability explainers.

**3. Background: Expert Systems & Chatbots**

An expert system encodes domain expertise to produce recommendations or diagnoses. Traditional systems use rule-based engines, but modern expert assistants often combine symbolic knowledge with data-driven models. Chatbots powered by Large Language Models (LLMs) can produce fluent explanations but may hallucinate; grounding them with retrieval (RAG) is an effective hybrid approach.

**4. Background: Retrieval-Augmented Generation (RAG)**

4.1 Core Concepts

- Retrieval: locate relevant documents or document chunks from a corpus given a user query. This typically uses vector similarity search on embeddings.
- Augmentation: supply the retrieved passages (context) to a generative model as evidence to condition its output.
- Generation: the LLM composes an answer using provided context and optionally its pre-trained knowledge.

  4.2 Operational Flow

1. Query encoding: convert the user query to an embedding vector using the same embedding model used for the corpus.
2. Similarity search: query the vector index (Pinecone) to return top-k relevant document chunks.
3. Prompt assembly: create a prompt or chain that injects retrieved chunks and system instructions to the LLM.
4. Generation: the LLM produces an answer constrained by the context and system prompt.
5. (Optional) Post-processing: trim, format, or include citations of the retrieved sources.

4.3 RAG in this Project

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384-d) via HuggingFaceEmbeddings.
- Vector store: Pinecone, index name `cybersecurity-advisor-chatbot`, Serverless spec for cloud.
- Retrieval: `docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})` returns top-3 chunks.
- Prompting: a strict system prompt is defined to force reliance on context and restrict out-of-scope questions. The project uses LangChain chains to combine retrieved docs and call the generation model.

Mathematically, retrieval is nearest-neighbor search in embedding space: given query vector q and corpus vectors {v_i}, find top-k by maximizing similarity s(q, v_i), typically cosine similarity.

$$ s\_{cos}(q, v) = \frac{q \cdot v}{\|q\|\,\|v\|} $$

This project sets `k=3` (or `k=5` in some comments) to balance context breadth vs prompt size.

**5. System Architecture and Components**

High-level components:

- PDF ingestion layer (`src/helper.py`) — loads PDFs, filters metadata, and splits content into chunks.
- Embedding layer (`download_hugging_face_embeddings`) — provides consistent embeddings for corpus and queries.
- Indexing layer (`store_index.py`) — creates Pinecone index and uploads chunk embeddings.
- Incremental updater (`store_new_data.py`) — adds ad-hoc documents to the existing index.
- Retrieval and RAG chain (`app.py`) — creates retriever, constructs prompt template, and invokes the generative model.
- Web UI (`templates/chat.html`, `static/style.css`) — user input and visual presentation.

Sequence of operations (normal run):

1. Prepare PDFs in `data/`.
2. Run `python store_index.py` to create Pinecone index and upload embeddings.
3. Start the web app with `python app.py` and open the UI.
4. User submits query; backend retrieves context and generates response.

**6. Data pipeline: PDF ingestion to vector index**

Major steps (implemented across `src/helper.py` and `store_index.py`):

- DirectoryLoader with `PyPDFLoader` loads all PDFs in `data/`.
- Documents are filtered to minimal metadata (source only) via `filter_to_minimal_docs`.
- `RecursiveCharacterTextSplitter` splits long pages into chunks (default chunk_size=500, overlap=20) producing smaller documents suitable for embedding.
- Embeddings are obtained using HuggingFace `all-MiniLM-L6-v2`.
- If the Pinecone index does not exist, `store_index.py` will create it with `dimension=384` and `metric='cosine'`.
- Chunks are upserted into Pinecone via `langchain_pinecone.PineconeVectorStore`.

**7. File-by-file code explanations**

7.1 `app.py` — Flask application and RAG chain

- Purpose: start a Flask web server, load environment variables, connect to Pinecone, build a retriever and RAG chain, and expose endpoints for the UI.

Key excerpts and explanation:

- Environment loading:
  - `load_dotenv()` reads `.env` for `PINECONE_API_KEY` and `GROQ_API_KEY`.
  - The code sets OS environment variables to ensure library clients can read them.

- Embeddings and docsearch:
  - `embeddings = download_hugging_face_embeddings()` (from `src.helper`) returns a `HuggingFaceEmbeddings` instance.
  - `PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)` attaches the existing Pinecone index as `docsearch`.

- Prompt and retrieval chain:
  - A strict `system_prompt` string forces the model to answer from provided context and decline out-of-scope asks.
  - `ChatPromptTemplate.from_messages(...)` builds the prompt with a system message and a human input slot.
  - `retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={'k': 3})` sets up retrieval.
  - `chatModel = ChatGroq(...)` initializes the Groq-backed chat model (Llama 3.1 8b instant in the code); it reads `GROQ_API_KEY`.
  - `question_answer_chain = create_stuff_documents_chain(chatModel, prompt)` creates a chain that stuffs all retrieved docs into the prompt and calls the model.
  - `rag_chain = create_retrieval_chain(retriever, question_answer_chain)` composes retrieval + QA chain.

- Routes:
  - `/` returns the chat UI template `chat.html`.
  - `/get` accepts POST requests (AJAX) with `msg` form field; it:
    - logs the input,
    - retrieves documents for debugging using `retriever.invoke(msg)` (the code prints retrieved chunks),
    - invokes `rag_chain.invoke({"input": msg})`, expects `response["answer"]` and returns it as string to the front-end.

- Notes:
  - `k` value controls how many chunks are passed; this affects prompt length.
  - The system prompt contains explicit identity and out-of-scope handling to limit hallucination.

  7.2 `src/helper.py`

- Purpose: data ingestion and embeddings helper functions.

Functions:

- `load_pdf_file(data)`
  - Uses `DirectoryLoader` (glob="\*.pdf") with `PyPDFLoader` to load PDFs from `data/` directory.
  - Returns a list of `Document` objects (LangChain `Document`).

- `filter_to_minimal_docs(docs: List[Document]) -> List[Document]`
  - Produces a sanitized list of `Document` objects that retain only `page_content` and `metadata['source']`.
  - This reduces index bloat and ensures consistent metadata for retrieval citation.

- `text_split(extracted_data)`
  - Splits documents into chunks using `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)`.
  - Returns chunked documents ready for embeddings.

- `download_hugging_face_embeddings()`
  - Instantiates `HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')`.
  - This ensures the same embedding model is used throughout indexing and querying.

  7.3 `src/prompt.py`

- Purpose: central place to store prompt templates used by the chains.
- Content: a `system_prompt` that asks the assistant to use at most three sentences, be concise, and rely on provided context. The `app.py` also defines another stricter prompt; both are examples of prompt engineering patterns.

  7.4 `store_index.py`

- Purpose: create a Pinecone index and upload embeddings for all documents in `data/`.

Flow:

1. Load environment variables and set `PINECONE_API_KEY` and `GOOGLE_API_KEY` (if present).
2. `extracted_data = load_pdf_file(data='data/')` — gather PDFs.
3. `filter_data = filter_to_minimal_docs(extracted_data)` — sanitize docs.
4. `text_chunks = text_split(filter_data)` — split to chunks.
5. `embeddings = download_hugging_face_embeddings()` — embeddings instance.
6. Create Pinecone client `pc = Pinecone(api_key=pinecone_api_key)`.
7. If index doesn't exist, `pc.create_index(name=index_name, dimension=384, metric='cosine', spec=ServerlessSpec(...))`.
8. `docsearch = PineconeVectorStore.from_documents(documents=text_chunks, index_name=index_name, embedding=embeddings)` — upsert chunks and build index.

Notes:

- `dimension=384` must match the embedding model chosen.
- `chunk_size` and `overlap` selection affect retrieval recall and context redundancy.

  7.5 `store_new_data.py`

- Purpose: add a new ad-hoc document into an existing Pinecone index (for project metadata or small updates).

Flow:

1. Load environment and embeddings, then connect to existing index via `PineconeVectorStore.from_existing_index(...)`.
2. Create a `Document` with `page_content` describing the project (authors, supervision) and `metadata={'source': 'Expert Systems'}`.
3. `docsearch.add_documents(documents=[info])` adds the doc into the index.
4. Wait 10 seconds for eventual consistency; test retrieval by querying `retriever.invoke(query)` and print results.

7.6 `check_models.py`

- Purpose: small utility to list Google generative models available for the provided `GOOGLE_API_KEY`.
- Behavior: configures `google.generativeai` and prints models that support `generateContent`.

  7.7 `templates/chat.html` and `static/style.css`

- `chat.html` provides a Bootstrap-based chat UI with:
  - A message area where user and bot messages are appended.
  - A credits modal listing department, supervisors, and team members.
  - Client-side JavaScript using jQuery to POST queries to `/get` and render responses.

- `static/style.css` styles the chat UI: dark theme, animated bot avatar, message containers, responsive layout.

  7.8 `requirements.txt`, `README.md`, `Dockerfile`, `setup.py`

- `requirements.txt` lists pinned or needed packages: `langchain`, `flask`, `sentence-transformers`, `pypdf`, `python-dotenv`, `langchain-pinecone`, `langchain-google-genai`, `langchain-community`, `langchain-groq`, plus editable install `-e .`.
- `README.md` contains setup steps: conda env creation, `pip install -r requirements.txt`, `.env` configuration, running `python store_index.py`, and `python app.py`.
- The repository includes a `Dockerfile` (not fully described in this document) and `setup.py` that support packaging; inspect and adapt them for production deployments.

**8. Deployment and running instructions**

Development (local):

- Create environment and install dependencies (as in `README.md`):

```bash
conda create -n medibot python=3.13.13 -y
conda activate medibot
pip install -r requirements.txt
```

- Create `.env` in project root with:

```
PINECONE_API_KEY="<your_pinecone_key>"
GROQ_API_KEY="<your_groq_key>"  # if using Groq
GOOGLE_API_KEY="<google_key>"  # if using google genai
```

- Build and populate index (first time or after adding PDFs):

```bash
python store_index.py
```

- (Optional) Add a brief project metadata document:

```bash
python store_new_data.py
```

- Run the Flask app:

```bash
python app.py
```

- Open `http://localhost:8080` in a browser and interact with the assistant.

Production notes:

- Use environment variables in a secure secrets store (GitHub Actions secrets, AWS Secrets Manager).
- For scale, run the Flask app under a WSGI server (gunicorn) behind a reverse proxy, and ensure Pinecone creds are valid.
- Use a container image and CI/CD pipeline to deploy to EC2 or ECS/EKS. Follow `README.md` deployment notes.

**9. Evaluation, testing, and validation**

Evaluation checklist:

- Retrieval QA: verify that the retriever returns documents that are truly relevant for a set of test queries.
- Grounding test: craft questions whose answers are present and absent in the corpus; confirm the assistant uses context when available and declines or uses minimal external knowledge when out-of-scope.
- Latency and prompt size: measure token usage and model latency for various `k` values; reduce `k` if prompt size causes timeouts.

Suggested tests (manual or automated):

- Create unit tests for `src/helper.py` functions (load_pdf_file, text_split).
- Create integration test: index a small set of known PDFs, query them, assert that the top result contains expected snippet.

**10. Limitations, risks and mitigation**

Limitations:

- Hallucination risk if retrieved context is insufficient.
- Biases and incorrect content inherited from source materials or LLM pretraining.
- Privacy: avoid indexing sensitive documents or PII.

Mitigations:

- Tune chunking and `k` to maximize retrieval of relevant evidence.
- Increase transparency: return citations and snippets with answers.
- Add a final verification stage (e.g., checking consistency with multiple retrieved docs).

**11. Future work and enhancements**

- Add citation markers in responses that reference which PDF and page/chunk answered the question.
- Add per-document metadata (title, author, page number) to improve traceability.
- Implement multi-turn memory: persist conversation history and use it to retrieve context.
- Add role-based access control and secure hosting.
- Improve prompt engineering and use advanced pipelines (rerankers, cross-encoders) for better retrieval.

**12. Conclusion**

This project demonstrates a practical RAG-based expert assistant tailored for cybersecurity education. By combining a curated, indexed knowledge base with a generative LLM and a strict system prompt, the system reduces hallucination and provides concise, context-backed answers appropriate for an academic environment.

**13. References and Appendix**

- LangChain documentation: https://langchain.readthedocs.io
- Pinecone docs: https://www.pinecone.io/docs/
- sentence-transformers: https://www.sbert.net/
- Project source files (for code-level inspection): see the repository root.

---

Appendix A — Quick file reference

- [app.py](app.py): Flask app and RAG chain entrypoint.
- [store_index.py](store_index.py): Create and populate Pinecone index from `data/` PDFs.
- [store_new_data.py](store_new_data.py): Add a single document to the existing index.
- [src/helper.py](src/helper.py): PDF loading, filtering, splitting, and embedding helper.
- [src/prompt.py](src/prompt.py): Prompt templates.
- [templates/chat.html](templates/chat.html): Front-end UI.
- [static/style.css](static/style.css): UI styling.
- [requirements.txt](requirements.txt): Python dependencies.
- [README.md](README.md): Quick-start and deployment notes.

---