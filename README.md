Document RAG Solution with LlamaIndex
Overview

This project provides a Retrieval-Augmented Generation (RAG) solution that allows users to ask questions across a collection of PDF, DOCX, and PPTX documents.

The implementation indexes documents using LlamaIndex, generates embeddings with OpenAI, and uses an OpenAI chat model to answer questions based on retrieved content.

Files

```ask_questions_about_reports.py``` - Main application that loads documents, builds the index, and answers questions.

```requirements.txt``` - Python dependencies.

```documents/``` - Folder containing PDF, DOCX, and PPTX files.

How It Works

* Load documents from the documents directory.
* Extract text from PDF, DOCX, and PPTX files.
* Create LlamaIndex documents.
* Generate embeddings using text-embedding-3-small.
* Build a vector index.
* Retrieve relevant document chunks.
* Generate answers using gpt-4o-mini.

Architecture

Documents
```
(PDF/DOCX/PPTX)
        |
        v
Text Extraction
        |
        v
LlamaIndex
        |
        v
OpenAI Embeddings
        |
        v
Vector Index
        |
        v
Retriever
        |
        v
GPT-4o-mini
        |
        v
Answer
```

Installation

Create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```
Configure OpenAI
Set your API key:

```
export OPENAI_API_KEY=your-api-key
```

Windows:

```
set OPENAI_API_KEY=your-api-key
```

Place documents in:

documents/

Run:

```
python src/ask_questions_about_reports.py
```
Example:

Question: What are the key project risks?
Question: Summarize the customer requirements.
Question: Which document mentions budget overruns?
Exit:

exit


Suggested Improvements

* Persist the index to disk.
* Add source citations to answers.
* Use CitationQueryEngine.
* Add support for MChat.
* Use ChromaDB or Qdrant for larger datasets.
Build a FastAPI or Streamlit interface.
License
Use and modify as needed for your organization.
