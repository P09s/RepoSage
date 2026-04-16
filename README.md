# RepoSage

RepoSage is a codebase intelligence tool that leverages Retrieval-Augmented Generation (RAG) to help developers understand and navigate repositories. By ingesting a GitHub repository and building a local vector index, RepoSage allows users to ask natural language questions about codebase architecture, data flows, dependencies, and logic.

## Features

* **Direct Repository Ingestion:** Clone and index public or private GitHub repositories directly from a URL.
* **Code-Aware Parsing:** Processes and chunks source code files specifically structured for optimal context retrieval.
* **Local Vector Search:** Utilizes HuggingFace embeddings and ChromaDB for fast, local vector storage and similarity search.
* **Intelligent Chat Interface:** Powered by LangChain and Groq (Llama 3.1) to synthesize accurate technical answers based on your code.
* **Source Attribution:** Responses include direct references to the specific files used to generate the answer.

## Tech Stack

* **Frontend:** Streamlit
* **LLM Orchestration:** LangChain
* **LLM Provider:** Groq (Llama 3.1)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace

## Getting Started

### Prerequisites

* Python 3.9 or higher
* A Groq API Key
* (Optional) A GitHub Personal Access Token for accessing private repositories

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/yourusername/RepoSage.git](https://github.com/yourusername/RepoSage.git)
   cd RepoSage
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your environment variables. Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Usage

1. Launch the application using Streamlit:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the provided local URL (typically `http://localhost:8501`) in your web browser.
3. In the sidebar, enter the target GitHub repository URL (and your access token if the repository is private).
4. Click **Load Repository** and wait for the indexing process to complete.
5. Once indexed, use the main chat interface to ask technical questions about the loaded codebase.

## Project Structure

* `app.py`: The main Streamlit application handling the UI and state management.
* `rag_engine.py`: Contains the logic for building the ChromaDB vector store and the LangChain RAG pipeline.
* `repo_ingestion.py`: Handles cloning the target GitHub repository, parsing the directories, and chunking the files for embedding.