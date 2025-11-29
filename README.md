# 📄 Local PDF Question-Answering App (Streamlit + Ollama)

A fully local **PDF Question-Answering** application built using **Streamlit** and **Ollama**.  
Upload a PDF → Process it → Ask questions → Get answers from a local LLM.  
Everything runs **offline**, **private**, and **on your own machine**, no cloud APIs, no external servers.

---

## ✨ Features

- 📁 Upload any PDF  
- 📄 Extract and read text from the PDF  
- ✂️ Split the text into chunks for retrieval  
- 🔍 Retrieve relevant chunks using keyword matching (no embeddings required)  
- 🤖 Use a local LLM (via Ollama) to answer questions  
- 🛡 100% offline and privacy-focused  
- 🧩 Clean, simple codebase — easy to modify  

---

## 🧰 Tech Stack

| Component       | Technology     |
|-----------------|----------------|
| UI              | Streamlit      |
| PDF Parsing     | PyPDF          |
| Retrieval       | Keyword Search |
| LLM Backend     | Ollama         |
| HTTP Requests   | Python Requests |

---

# 📦 Installation Guide

## 1️⃣ Install Python 3.10+

Download Python for Windows:  
➡️ https://www.python.org/downloads/windows/

During installation, enable:

✔ **Add Python to PATH**

---

## 2️⃣ Install Ollama

Download Ollama for Windows/macOS/Linux:  
➡️ https://ollama.com/download

Then pull the model:

```bash
ollama pull llama3.2
```

## 3️⃣ Install Required Python Packages

Run these in PowerShell or CMD:

```bash
pip install --upgrade pip
pip install streamlit pypdf requests
```

➡️ No virtual environment is required.

## ▶️ Running the Application

To run the application, navigate to the folder containing `app.py` and execute:

```bash
streamlit run app.py
```

The app will open automatically at:

```bash
http://localhost:8501
```

## 🧠 How It Works (Simple RAG)

This project implements a lightweight form of **Retrieval-Augmented Generation (RAG)**:

### 1. PDF Extraction
The PDF is parsed using **PyPDF** to extract raw text.

### 2. Chunking
The extracted text is split into overlapping chunks:
- **800 characters per chunk**
- **200 characters overlap**  
This helps maintain context continuity across chunks.

### 3. Keyword Indexing
Each chunk is tokenized into a set of **keywords**.  
These keyword sets form a simple, fast index for matching user queries.

### 4. Retrieval
When a user asks a question:
- The question is tokenized.
- Each chunk is scored based on **keyword overlap**.
- The **top-matching chunks** are selected and fed to the LLM to generate the final answer.

### 5. Local LLM Answering
Selected chunks + the user’s question are sent to:
```bash
http://localhost:11434/api/chat
```
(using the locally running Llama model)
The model responds using only the provided context.

## 🛠 Future Enhancements

This project can be extended with several powerful upgrades:

- 🔹 **Embeddings-based retrieval** (e.g., BGE, MiniLM, nomic-embed-text)
- 🔹 **Vector search** using FAISS, Chroma, or SQLite-based vector indexes
- 🔹 **LangChain-powered RAG pipeline** for modular orchestration
- 🔹 **Chat history panel** to maintain context across turns
- 🔹 **Multi-PDF support** for larger document collections
- 🔹 **Dark / Light themes** for better accessibility
- 🔹 **Improved UI styling** for a smoother user experience
