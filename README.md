# 🌾 Multimodal Crop Health Agent

An interactive **Agentic RAG** application that uses **Grounding DINO** and **CLIP** for visual crop analysis and **LlamaIndex** with **Gemini** to retrieve treatment protocols from PDF manuals.

## 🚀 Setup Instructions

### 1. Environment Preparation

**Python 3.10+** installed.

```bash
# Create a virtual environment
python3 -m venv .venv
# or
uv venv .

# Activate the environment
source .venv/bin/activate
```

### 2. Install Dependencies

Install all required packages, including LlamaIndex, Pydantic AI, and Streamlit.

```bash
uv sync
```

### 3. Configure API Keys

Create a `.env` file in the root directory and add your credentials.

```env
GOOGLE_API_KEY=<...>
LLAMA_CLOUD_API_KEY=<...>
HF_TOKEN=<...>
```

---

## 🛠️ How to Run

### Step 1: Ingest Manuals (RAG Engine)

Before running the app, you must build the vector database. Place your PDF manuals in the `data/manuals/` folder, then run the ingestion script.

```bash
# Ingest all manuals in the directory
python -m app.rag.ingest --target ./data/manuals/

# Or ingest a specific file
python -m app.rag.ingest --target ./data/manuals/abc.pdf
```

*This script uses **LlamaParse** to accurately extract tables and **Metadata Filtering** to tag documents by crop type.*

### Step 2: Launch the Streamlit App

Once the database is ready, start the web interface.

```bash
dotenv run python3 -m streamlit run app/streamlit_app.py
# or
uv run -m streamlit run app/streamlit_app.py
```

---

## 📖 Project Structure

* `app/agent/`: Core agent logic and tools.
* `app/rag/`: Ingestion and query engine scripts.
* `app/vision/`: Image processing logic (DINO/CLIP).
* `data/`: Local storage for PDFs and the ChromaDB vector store.

---
