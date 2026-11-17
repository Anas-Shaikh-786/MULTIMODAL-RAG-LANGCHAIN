# 🤖 Multimodal RAG LangChain

A modular, enterprise-grade **Multimodal Retrieval-Augmented Generation (RAG)** framework built with LangChain. It combines advanced multimodal document ingestion (`Smart_Document_Loader`) with an end-to-end RAG execution pipeline (`RAG`) and an interactive, user-authenticated **Streamlit Web Application**.

---

## 🌟 Key Features

### 📄 Smart Multimodal Document Loader (`Smart_Document_Loader`)
* **Multi-Format Support**: PDFs, Images (PNG, JPG, JPEG), Word (DOCX), Text (TXT, MD), Data (CSV, JSON), and Archives.
* **Layout & Structure Analysis**: Layout detection powered by `DocLayout-YOLO` and table structure recognition using `TATR`.
* **OCR & VLM Processing**: Text and OCR correction via EasyOCR/PaddleOCR, figure understanding & formula extraction powered by Vision-Language Models (**Google Gemini**).
* **Semantic Chunking**: Element-aware semantic chunking with configurable target chunk size and overlap settings.

---

### 🧠 Advanced RAG Execution Framework (`RAG`)
* **Hybrid Retrieval Strategies**: `hybrid` (Reciprocal Rank Fusion of Similarity + BM25), `mmr` (Maximal Marginal Relevance), `similarity`, and `bm25`.
* **Reranking Engine**: Cross-Encoder reranking (`bge-reranker-base`, `bge-reranker-large`, `ms-marco`).
* **Query Rewriting**: Contextual query expansion and transformation using LLMs.
* **Multi-Turn Conversational Memory**:
  * **Memory Types**: `conversation`, `summary`, `entity`, and `hybrid` memory.
  * **Persistent Storage**: Filesystem & MongoDB memory store backends.
* **Grounded Citations**: Automatic source citation extraction with Document Name, Page Number, and Section Title.

---

### ✨ User-Authenticated Streamlit Web Application (`app.py`)
* **Role-Based Access & Authentication**: Multi-user login and registration system (`auth.py`).
* **Interactive Knowledge Studio**: Chat assistant with real-time streaming, collapsible citations, and query rewriting transparency.
* **Document Management Center**: Upload and process documents on-the-fly into the active vector index.
* **Memory Inspector**: Inspect live conversation turns, summaries, and extracted entities for the logged-in user.
* **Admin Command Center**: User account management and storage telemetry.

---

## 📂 Project Architecture

```text
Multimodal_RAG_LangChain/
├── app.py                      # Streamlit Web Application entry point
├── auth.py                     # User authentication & credentials module
├── usage.py                    # Application launcher script
├── RAGPipeline.ipynb           # Interactive notebook pipeline demo
├── requirements.txt            # Python dependencies
│
├── Smart_Document_Loader/      # Ingestion & document parsing engine
│   ├── loader.py               # Main SmartDocumentLoader class
│   ├── extraction/             # Text, Table, Figure, and Formula extractors
│   ├── layout/                 # Layout detection wrappers
│   ├── chunking/               # Semantic chunking logic
│   └── chunk_store/            # Document chunk serializers
│
├── RAG/                        # RAG Execution Pipeline
│   ├── pipeline.py             # Main RAGPipeline orchestrator
│   ├── retrievers/             # Hybrid, MMR, Similarity, & BM25 retrievers
│   ├── rerankers/              # Cross-Encoder rerankers
│   ├── memory/                 # Conversation, Summary, Entity, & Hybrid memory
│   ├── llms/                   # LLM provider wrappers (Groq)
│   ├── embedings/              # HuggingFace & provider embedding factories
│   ├── vector_stores/          # FAISS, Chroma, Milvus, Pinecone adapters
│   └── query_rewriting/        # Query rewriting modules
│
└── storage/                    # Persistent storage directory
    ├── raw_documents/          # Original uploaded files per user
    ├── processed_documents/    # Saved document chunks (.sdl)
    ├── memory_store/           # User memory JSON state files
    └── users.json              # Registered user credentials database
```

---

## ⚙️ Prerequisites & Setup

### 1. Environment Setup
Make sure Python **3.11+** is installed. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Linux / macOS
```

### 2. Install Dependencies
Install all required dependencies from `requirements.txt` along with `streamlit`:

```bash
pip install -r requirements.txt
pip install streamlit
```

### 3. API Keys Configuration
Create a `.env` file in the project root directory and add your API keys:

```env
GROQ_API_TOKEN=your_groq_api_key_here
GEMINI_API_KEY=your_google_gemini_api_key_here
```

---

## 🚀 How to Use

### Method 1: Launch the Streamlit Web Application

Run the Streamlit application directly:
```bash
streamlit run app.py
```
Or run the launcher script:
```bash
python usage.py
```
Open your browser at `http://localhost:8501`.

---

### Method 2: Python Code Usage

#### 1. Ingest Documents using `SmartDocumentLoader`:

```python
import os
from dotenv import load_dotenv
from Smart_Document_Loader.loader import SmartDocumentLoader

load_dotenv()

# Initialize Document Loader
loader = SmartDocumentLoader(
    llm_config={
        "provider": "groq",
        "api_key": os.getenv("GROQ_API_TOKEN")
    },
    vlm_config={
        "provider": "gemini",
        "api_key": os.getenv("GEMINI_API_KEY")
    },
    storage={
        "root": "storage",
        "raw_documents_dir": "raw_documents",
        "metadata": {"type": "filesystem"},
        "chunks": {
            "type": "filesystem",
            "output_directory": "processed_documents",
        },
    }
)

# Ingest document
document_id, document_chunks = loader.invoke("path/to/your/document.pdf")
print(f"Ingested document ID: {document_id}, Total chunks: {len(document_chunks)}")
```

#### 2. Query Knowledge using `RAGPipeline`:

```python
from RAG.pipeline import RAGPipeline

# Initialize RAG Pipeline
rag = RAGPipeline(
    user_id="user_123",
    storage={
        "root": "storage",
        "raw_documents_dir": "raw_documents",
        "metadata": {"type": "filesystem"},
        "chunks": {
            "type": "filesystem",
            "output_directory": "processed_documents",
        },
    },
    embedding_config={
        "provider": "huggingface",
        "model": "bge-base"
    },
    vectorstore_config={
        "type": "faiss"
    },
    retriever_config={
        "type": "hybrid",
        "k": 5,
        "candidate_k": 20,
        "lambda_mult": 0.5,
        "rrf_k": 60,
    },
    reranker_config={
        "type": "cross-encoder",
        "model": "bge-reranker-base"
    },
    context_builder_config={
        "type": "default",
        "include_metadata": True,
        "include_source": True,
        "include_page_number": True,
        "include_section_title": True,
    },
    prompt_builder_config={
        "type": "default",
        "response_style": "brief",
        "system_prompt": "You are a knowledgeable AI assistant.",
        "instructions": "Answer using the provided context and available conversation memory only.",
        "use_context": True,
    },
    llm_config={
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "api_key": os.getenv("GROQ_API_TOKEN"),
        "temperature": 0.1,
        "max_tokens": 1024,
    },
    query_rewriting_config={
        "enable": True
    },
    citation_builder_config={
        "type": "default",
        "include_document": True,
        "include_page_number": True,
        "include_section_title": True,
    },
    memory_config={
        "type": "conversation",
        "max_turns": 3,
        "store": {
            "type": "filesystem"
        }
    }
)

# Add ingested chunks to the pipeline index
rag.add_documents(document_chunks)

# Ask a question
response = rag.ask("What is the main architecture discussed in the paper?")

print("Answer:", response.answer)
print("Citations:", response.citations)
```
