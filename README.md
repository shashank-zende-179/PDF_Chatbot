# 📄 PDF-Based RAG Chatbot System

An AI-powered document question-answering system that leverages **Retrieval-Augmented Generation (RAG)** to deliver accurate, context-aware responses from one or multiple PDF documents using semantic search and Large Language Models.

---

## 🏗️ Architecture & Tech Stack

This project follows a classic machine-learning **RAG architecture**, decoupling the UI from the heavy data processing and model inferencing logic.

### Tech Stack:
* **Frontend**: [Streamlit](https://streamlit.io/) (Fast, python-native UI framework)
* **Backend API**: [FastAPI](https://fastapi.tiangolo.com/) (High-performance, async web framework)
* **PDF Processing**: [PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/) (Extremely fast C-based PDF extractor)
* **Vector Embeddings**: [SentenceTransformers](https://sbert.net/) (Using `all-MiniLM-L6-v2` runs locally and produces 384-dimensional vector embeddings without incurring API costs)
* **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search, a highly optimized localized vector store)
* **LLM Generation**: [Groq API](https://groq.com/) (Using the lightning-fast `llama-3.1-8b-instant` model to synthetically generate text strictly from prompt contexts).

  ```

User

   │

   ▼

Upload PDF Documents

   │

   ▼

PyMuPDF Text Extraction

   │

   ▼

Text Cleaning & Chunking

   │

   ▼

SentenceTransformer Embeddings

   │

   ▼

FAISS Vector Database

   │

─────────────────────────────────────────────

              User Question

                     │

                     ▼

          Query Embedding Generation

                     │

                     ▼

        Semantic Similarity Search

                     │

                     ▼

      Top Relevant Document Chunks

                     │

                     ▼

      Groq Llama 3.1 (LLM)

                     │

                     ▼

          Context-Aware Response

                     │

                     ▼

            Streamlit Interface

```






---

## 🗂️ Project Structure & File Index

```text
chat/
├── app/
│   ├── __init__.py         # Python package initiator
│   ├── main.py             # FastAPI entry point & API Controller
│   ├── pdf_processor.py    # Document interaction layer (parsing & chunking logic)
│   ├── embedding.py        # Local machine learning embedding handler
│   ├── retriever.py        # Vector mapping and database indexing instance
│   └── llm.py              # Generative AI interface (prompt assembly & API request)
├── ui/
│   └── streamlit_app.py    # The graphical user interface framework configuration
├── requirements.txt        # All python dependencies required for virtual environments
└── README.md               # This documentation file
```

---

## ⚙️ Core Data Flow Framework

The following flowchart details the exact computation pipeline when a user interacts with the system.

### 1. The Upload Pipeline (`/upload` endpoint routing)
1. **User Action**: The user selects a document via the Streamlit UI and clicks "Process PDF".
2. **Transfer**: Streamlit's multi-part payload system sends the raw PDF to the FastAPI backend limitlessly.
3. **Extraction**: `app.pdf_processor.py` intercepts the PDF, utilizes PyMuPDF, and extracts text page-by-page. It forcefully appends `[Page N]` tags to all text dynamically.
4. **Chunking**: The extracted text is passed to an algorithm that splits sentences into chunks of exactly ~500 words, intentionally overlapping them by 100 words. (Overlapping ensures sentences split midway do not lose their semantic relevance to neighboring concepts).
5. **Embedding**: `app.embedding.py` runs those textual chunks through the local HuggingFace `SentenceTransformer` downloading numerical representations (floating-point tensors) defining the implicit semantic meaning of those paragraphs.
6. **Indexing**: `app.retriever.py` appends those floating-point block embeddings into the local `FAISS` database hierarchy map. It then inherently saves `faiss_index.bin` and `chunks.pkl` to the local hard drive to remember this PDF system context permanently.

### 2. The Request Pipeline (`/ask` endpoint routing)
1. **User Action**: The user types a bespoke question into the active text field in the UI interface.
2. **Transfer**: Streamlit crafts an isolated POST request attaching the query string and your explicit Groq API key via secure sockets.
3. **Query Embedding**: The exact initial `SentenceTransformer` module tokenizes the user's explicit question into its relevant numerical map form.
4. **Context Retrieval**: FAISS mathematically calculates and retrieves the closest `Top-3` textual chunks that exhibit the shortest Euclidean distance against the user's query meaning map.
5. **LLM Synthesis**: `app.llm.py` structurally groups the user's specific query explicitly bounded against those `Top-3` context chunks into a master programmatic prompt block, and securely dispatches it to the Groq Cloud infrastructure. (We enforce a rigid `temperature=0` requirement to isolate hallucinations alongside an explicit escape clause of `"Not Found"` text).
6. **Delivery**: The synthesized text block securely bounces back up to FastAPI logic, mapping to Streamlit frontend logic mapping, generating a dynamic dialog box alongside interactive nested toggle arrays displaying verbatim explicit context origins utilized.

---

---
# ⚙️ Installation
### Clone Repository
```bash
git clone https://github.com/yourusername/PDF-RAG-Chatbot.git
cd PDF-RAG-Chatbot
```
### Create Virtual Environment
```bash
python -m venv venv
```
### Activate Environment
**Windows**
```bash
venv\Scripts\activate
```
**Linux / macOS**
```bash
source venv/bin/activate

```
### Install Dependencies

```bash
pip install -r requirements.txt

```
### Configure Environment Variables

Create a `.env` file.
```env
GROQ_API_KEY=YOUR_API_KEY

```
### Run Backend
```bash
uvicorn backend:app --reload

```
### Run Frontend
```bash
streamlit run app.py
```
---

##  Future Enhancements

* OCR support for scanned PDFs
* Image and table understanding
* Voice interaction
* Multilingual support
* Hybrid search (Semantic + Keyword)
* Citation-aware responses
* Conversation memory
* User authentication
* Cloud deployment (AWS, Azure, GCP)
* Enterprise document integration

# 👨‍💻 Author

**Shashank Zende
