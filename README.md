# AmbedkarGPT — Local RAG Question‑Answering System

## 📌 About the Project
AmbedkarGPT is a **local Retrieval-Augmented Generation (RAG)** system that allows users to ask questions grounded strictly in a reference document.  
It loads text, splits it into chunks, builds a Chroma vector database, retrieves the most relevant chunks based on the query, and then generates an answer using a locally running LLM through **Ollama**.

### ✔ What this project does
- Loads and processes `speech.txt`
- Builds a vector database using **ChromaDB**
- Retrieves the top‑k most relevant chunks for any query
- Generates a **grounded** answer using a local LLM
- Automatically supports clean exit using commands like `quit` or `exit`

### ✔ Technologies used
- **Python 3.10+**
- **ChromaDB** (vector store)
- **Sentence‑Transformers** (embeddings)
- **Ollama** (local LLM runner)
- **Mistral (via Ollama)** for text generation
- **Torch**, **Transformers**, **LangChain Community modules**

---

# Repository structure

```
AmbedkarGPT/
│
├── main.py               # python code to create embeddings, Chroma vectorDB & answer generation (using Ollama)
├── speech.txt            # input document to index
├── chroma_db/            # generated vector DB folder
├── requirements.txt      # python dependencies
└── README.md
```
----

# 🚀 Installation & Usage Guide

Follow these steps to install and run the project locally.

---

## **1️⃣ Pull the repository**

```bash
git clone https://github.com/Deven10103/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

---

## **2️⃣ Create a virtual environment**

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## **3️⃣ Install dependencies**

```bash
pip install -r requirements.txt
```

---

## **4️⃣ Install Ollama**
Download Ollama from the official site:  
**https://ollama.com/download**

---

## **5️⃣ Pull the Mistral model**

```bash
ollama pull mistral
```

---

## **6️⃣ Run the main script**

```bash
python main.py
```

---

## ✔ Usage Instructions

After running the script, you will see a prompt like:

```
Enter your query:
```

- Type any question based on the document.
- If you want to exit the program, simply type:

```
quit
```
or
```
exit
```

---

# ⚠️ Important Note on GPU / VRAM Errors

If you face **model loading errors**, such as:

- `cudaMalloc failed`
- `failed to load model`
- `unable to allocate CUDA buffer`
- or any other GPU memory–related error

➡ **Close all heavy applications**, especially:
- web browsers (Chrome / Edge / Firefox)
- VS Code / JetBrains tools
- games or GPU‑accelerated programs

These may be using GPU VRAM, preventing the model from loading.  
Closing them usually resolves the issue.

---

# Quick debug commands

```powershell
nvidia-smi             # check GPU VRAM & processes
ollama list            # list local models
ollama run mistral   # test model
```

---


If you need modifications or want to extend the README, feel free to ask!
