# 🧠 RAG-Chatbot — Chat with Your Own Documents

This project allows you to build and run a **fully local Retrieval-Augmented Generation (RAG)** chatbot capable of understanding and answering questions based on your own PDF files or web content. You can query the system in natural language, and it will generate responses derived directly from your documents.

---

## 🧩 Technology Stack

* **LangChain** – For orchestration and prompt management
* **FAISS** – For high-speed semantic vector search
* **Ollama** – To run open-source Large Language Models (LLMs) locally
* **Streamlit** – For an intuitive, chat-based web interface

---

## ✨ Features

* 📄 Upload one or more PDFs or URLs as data sources
* 🧠 Convert documents into embeddings stored in a FAISS vector database
* 🔍 Retrieve contextually relevant information using semantic similarity search
* 💬 Generate accurate, context-aware answers using a local LLM
* 💻 100% local execution — no data leaves your system

---

## 🚀 Getting Started

### 1. Install Ollama and Run a Local LLM

Ensure that [Ollama](https://ollama.com) is installed on your system and download a compatible model such as `granite3.3`.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

#### Pull the Model

```bash
ollama pull granite3.3
```

#### Start the Model

```bash
ollama run granite3.3
```

---

### 2. Clone This Repository

```bash
git clone https://github.com/chandkund/RAG-Chatbot-Chat-with-Your-Own-Documents
cd RAG-Chatbot-Chat-with-Your-Own-Documents
```

---

### 3. Install Python Dependencies

Ensure you have **Python 3.9+** installed.

```bash
pip install -r requirements.txt
```

---

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 📂 How to Use

1. Launch the Streamlit app.
2. Use the sidebar to upload your PDF files or enter URLs.
3. Ask questions in the chat box using natural language.
4. The chatbot retrieves relevant sections and generates a response based on document context.

---

## 🏗️ Architecture Overview

* **Chat UI:** Built with Streamlit for real-time user interaction and file uploads.
* **LLMRAGHandler:** The core engine powered by LangChain. It handles retrieval, prompt creation, LLM interaction, and conversation memory.
* **Vector Store:** Uses FAISS to store embeddings and perform efficient similarity searches.
* **LLM (Ollama):** Runs locally using the `Granite 3.3` model, ensuring full data privacy and control.
* **Conversation Store:** Maintains conversation history locally (in JSON format) to support session continuity.

---

## ⚠️ Limitations

* Large documents may take time to process and embed.
* Response speed varies depending on the selected LLM model.
* Currently lacks quantitative evaluation metrics for generated answers.
* Designed primarily for local usage; cloud deployment not included yet.

---

## 💡 Future Improvements

* Implement **agentic RAG** (history-aware retrievers and dynamic tool-calling)
* Add support for additional data sources (e.g., Google Drive, Notion)
* Introduce document summarization and advanced UI features
* Enable cloud or hybrid deployment

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

