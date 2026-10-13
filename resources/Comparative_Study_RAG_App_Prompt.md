# Comparative Study Assessment RAG Application

## Objective
Build a production-quality Streamlit application that compares two studies against a set of assessment criteria using Retrieval-Augmented Generation (RAG).

### Requirements
1. Accept a folder containing documents for Study 1.
2. Accept a folder containing documents for Study 2.
3. Accept a text file containing assessment criteria.
4. Parse PDF, DOCX, and PPTX documents.
5. Build a separate LlamaIndex vector index for each study.
6. Use RAG to evaluate each study against every criterion.
7. Generate evidence-based scores and rationales.
8. Compare the two studies.
9. Display results through an interactive Streamlit dashboard.
10. Persist vectors locally using SimpleVectorStore.

## Technology Stack
- Python 3.12+
- Streamlit
- LlamaIndex
- OpenAI or Azure OpenAI
- SimpleVectorStore
- Plotly
- Pandas
- PyMuPDF
- python-docx
- python-pptx

## Supported Documents
- PDF
- DOCX
- PPTX

Create LlamaIndex documents with metadata:

```python
Document(
    text=content,
    metadata={
        "study": "...",
        "filename": "...",
        "filetype": "..."
    }
)
```

## RAG Architecture
Create one independent index per study:

```python
study1_index
study2_index
```

Do not merge indexes.

## Index Creation
- Chunk size: 1000
- Chunk overlap: 200
- VectorStoreIndex
- SimpleVectorStore

Persist under:

```text
storage/
  study1/
  study2/
```

Load persisted indexes when available.

## Retrieval
For each criterion:
- Retrieve evidence from Study 1
- Retrieve evidence from Study 2
- similarity_top_k=5

Capture:
- text
- filename
- relevance score

## Assessment Rubric
0-20: No evidence
21-40: Weak evidence
41-60: Partial evidence
61-80: Strong evidence
81-100: Comprehensive evidence

## Assessment Prompt
```text
You are an expert reviewer evaluating a scientific, clinical, regulatory, or research study.

You will receive:
1. An assessment criterion.
2. Evidence retrieved from study documents.

Evaluate how well the study addresses the criterion.

Provide:
- score
- strengths
- weaknesses
- rationale

Base your evaluation only on the supplied evidence.
Do not invent information.
Return valid JSON only.
```

Expected output:

```json
{
  "score": 82,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "rationale": "..."
}
```

## Comparison Prompt
```text
You are comparing two studies against the same criterion.

Given the assessments below:
- Study 1 Assessment
- Study 2 Assessment

Determine:
- winner
- confidence score
- explanation

Return JSON only.
```

## Dashboard
### Sidebar
- Study 1 Folder
- Study 2 Folder
- Criteria File
- Model Selection
- Run Assessment

### Visualizations
- Overall scores
- Radar chart
- Side-by-side bar chart
- Detailed criterion view

### Traceability
Show:
- source filename
- retrieved passage
- similarity score

## Persistence
Save:

```text
results/assessment_results.json
results/assessment_results.csv
```

## Project Structure
```text
study_assessor/
├── app.py
├── modules/
│   ├── document_loader.py
│   ├── index_manager.py
│   ├── retriever.py
│   ├── assessor.py
│   ├── dashboard.py
│   ├── prompts.py
│   ├── models.py
│   └── utils.py
├── storage/
├── results/
├── logs/
├── requirements.txt
└── README.md
```

## Quality Requirements
- SOLID principles
- Type hints
- Docstrings
- Modular design
- Production-ready implementation
- Complete runnable code
