# 🧠 Maternal Health AI Assistant

A Streamlit application for extracting knowledge from PDF documents pertaining to maternal healthcare data from the Pakistan National Health Services, and answering natural language queries using a combination of vector similarity and graph-based retrieval. The system also supports evaluating answer quality using ROUGE and BLEU scores.

---

## 🚀 Features

- 📄 Upload PDF documents
- 🔍 Extract text and split into chunks
- 🧠 Generate OpenAI embeddings and store in a Chroma vector database
- 🕸 Build a knowledge graph with named entities using spaCy and NetworkX
- 🧾 Ask questions about the documents using OpenAI's GPT models
- 🔄 Combine vector similarity and graph-based retrieval for enhanced answers
- 📈 Evaluate responses using **ROUGE** and **BLEU** metrics
- 🌐 Interactive knowledge graph visualization with **PyVis**
- 📊 Track query response times and log history

---

## 🧰 Technologies Used

- **Streamlit** for web UI
- **LangChain** for document splitting, embeddings, and LLM handling
- **ChromaDB** for vector storage
- **spaCy** for entity recognition
- **NetworkX** and **PyVis** for knowledge graph generation and visualization
- **OpenAI API (GPT-4-Turbo)** for answering questions
- **ROUGE** and **BLEU (nltk)** for answer evaluation
- **Matplotlib / Pandas** for displaying performance

---

## 🛠 Setup Instructions

1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-username/pdf-kg-qa.git
   cd pdf-kg-qa


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

