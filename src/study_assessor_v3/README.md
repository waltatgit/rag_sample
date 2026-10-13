# Comparative Study Assessment RAG Application

A Streamlit RAG tool designed to evaluate and compare two scientific/clinical/regulatory study document collections against assessment criteria using LlamaIndex and OpenAI models.

## Features
- **Document Support**: Parses `.pdf`, `.docx`, and `.pptx` documents with metadata.
- **Isolated Indexing**: Maintains two distinct LlamaIndex `VectorStoreIndex` vector stores saved under `storage/study1` and `storage/study2`.
- **RAG Evaluation**: Retrieves top-k evidence per criterion and generates JSON evaluations.
- **Side-by-Side Dashboards**: Interactive metrics, comparative radar charts, bar charts, and trace matrices.

## Quickstart

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt