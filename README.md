# Knowledge Graph QA with PDFs

A Streamlit application that builds a knowledge graph from PDF documents and provides question answering capabilities using LangChain and OpenAI.

## Features

- PDF document processing and text extraction
- Entity extraction using spaCy
- Knowledge graph construction with NetworkX
- Vector embeddings with OpenAI
- Natural language question answering
- Interactive knowledge graph visualization

## Prerequisites

- Python 3.10 or higher
- Anaconda or Miniconda (recommended)
- OpenAI API key

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/am07127/genai_knowledgegraph.git
cd genai_knowledgegraph
```

### 2. Create and activate conda environment

```bash
conda create -n kgrag_env python=3.10
conda activate kgrag_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
```

### 4. Set Up Environment Variables

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 5. Launch the streamlit app

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 6. Run the application

```bash
streamlit run app.py
```

