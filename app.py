import streamlit as st
from dotenv import load_dotenv
import os
import networkx as nx
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from pyvis.network import Network
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import tempfile
from streamlit.components.v1 import html
import nltk
from rouge import Rouge
import matplotlib.pyplot as plt
import pandas as pd
import time

# Download necessary NLTK resources
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

# Initialize the app with a title and icon
st.set_page_config(page_title="PDF Knowledge Graph QA", page_icon="🧠")

# Sidebar for settings and info
with st.sidebar:
    st.title("PDF Knowledge Graph QA")
    st.markdown("""
    This app allows you to:
    - Upload PDF documents
    - Extract knowledge into a vector database
    - Build a knowledge graph of entities
    - Query the combined knowledge with natural language
    - Evaluate system performance with ROUGE metrics
    """)
    
    # API key input
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit, LangChain, and OpenAI")

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.vectordb = None
    st.session_state.graph = None
    st.session_state.nlp = None
    st.session_state.uploaded_files = []
    st.session_state.performance_metrics = []
    st.session_state.ground_truth = {}
    st.session_state.response_history = []

# Main app
st.title("PDF Knowledge Graph Question Answering")
st.write("Upload PDF documents, then ask questions about their content.")

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

def initialize_system(uploaded_files):
    """Initialize the system with uploaded PDFs"""
    with st.spinner("Initializing system..."):
        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Load documents
        loader = DirectoryLoader(temp_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        st.success(f"Loaded {len(documents)} documents and split into {len(chunks)} chunks")
        
        # Initialize embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Initialize spaCy for entity extraction
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            st.warning("spaCy model 'en_core_web_sm' not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
        
        # Create a knowledge graph
        G = nx.Graph()
        
        # Extract entities and relationships
        for i, chunk in enumerate(chunks):
            doc = nlp(chunk.page_content)
            
            # Extract entities
            entities = [ent.text for ent in doc.ents]
            
            # Add nodes to graph
            for entity in entities:
                if entity not in G.nodes:
                    G.add_node(entity, type="entity")
            
            # Connect entities that appear in the same chunk
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    if G.has_edge(entity1, entity2):
                        G[entity1][entity2]['weight'] += 1
                    else:
                        G.add_edge(entity1, entity2, weight=1, type="co-occurrence")
            
            # Also connect entities to document chunks
            chunk_id = f"chunk_{i}"
            G.add_node(chunk_id, type="chunk", content=chunk.page_content)
            for entity in entities:
                G.add_edge(entity, chunk_id, type="appears_in")
        
        # Save to session state
        st.session_state.vectordb = vectordb
        st.session_state.graph = G
        st.session_state.nlp = nlp
        st.session_state.initialized = True
        st.session_state.uploaded_files = [f.name for f in uploaded_files]
        
        st.success("System initialized successfully!")

def enhanced_retrieval(query, top_k=3):
    """Enhanced retrieval combining vector search and knowledge graph"""
    if not st.session_state.initialized:
        return []
    
    # Step 1: Vector retrieval
    vector_results = st.session_state.vectordb.similarity_search(query, k=top_k)
    
    # Step 2: Entity extraction from query
    query_doc = st.session_state.nlp(query)
    query_entities = [ent.text for ent in query_doc.ents]
    
    # Step 3: Graph-based retrieval
    graph_results = []
    for entity in query_entities:
        if entity in st.session_state.graph.nodes:
            # Get connected chunks
            for neighbor in st.session_state.graph.neighbors(entity):
                if st.session_state.graph.nodes[neighbor].get('type') == 'chunk':
                    graph_results.append({
                        'content': st.session_state.graph.nodes[neighbor]['content'],
                        'score': st.session_state.graph[entity][neighbor].get('weight', 1)
                    })
    
    # Sort graph results by relevance score
    graph_results = sorted(graph_results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    # Step 4: Combine results
    combined_results = vector_results
    for result in graph_results:
        # Convert graph results to same format as vector results
        if result['content'] not in [doc.page_content for doc in combined_results]:
            combined_results.append(result['content'])
    
    return combined_results[:top_k]

def calculate_rouge_scores(reference, hypothesis):
    """Calculate ROUGE scores between reference and hypothesis texts"""
    if not reference or not hypothesis:
        return None
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]  # Return the first (and only) score
    except Exception as e:
        st.error(f"Error calculating ROUGE: {str(e)}")
        return None

def answer_question(query):
    """Answer a question using the enhanced retrieval system"""
    if not st.session_state.initialized:
        return "System not initialized. Please upload PDF files first."
    
    try:
        start_time = time.time()
        
        # Get enhanced context
        context_docs = enhanced_retrieval(query)
        
        # Combine context into a single string
        context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else doc for doc in context_docs])
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4-turbo")
        
        # Create prompt with context
        prompt = f"""
        Context information:
        {context}
        
        Question: {query}
        
        Answer the question based on the context provided. If you cannot answer from the context, state that clearly.
        Be concise and to the point. If medical information is requested, always include a disclaimer that this is not professional medical advice.
        """
        
        # Generate response
        response = llm.invoke(prompt)
        answer = response.content
        
        # Calculate time taken
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Log response for performance tracking
        st.session_state.response_history.append({
            'query': query,
            'answer': answer,
            'time_taken': time_taken,
            'timestamp': time.time()
        })
        
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"

def visualize_graph():
    """Visualize the knowledge graph"""
    if not st.session_state.initialized or st.session_state.graph is None:
        st.warning("Please initialize the system with PDF files first.")
        return
    
    with st.spinner("Generating graph visualization..."):
        # Create a PyVis network
        net = Network(height="600px", width="100%", notebook=True, directed=False)
        
        # Add nodes and edges
        for node in st.session_state.graph.nodes():
            node_type = st.session_state.graph.nodes[node].get('type', 'unknown')
            color = "#97c2fc" if node_type == "entity" else "#ffb347"
            net.add_node(node, color=color, title=node_type)
            
        for edge in st.session_state.graph.edges():
            weight = st.session_state.graph.edges[edge].get('weight', 1)
            edge_type = st.session_state.graph.edges[edge].get('type', 'unknown')
            net.add_edge(edge[0], edge[1], value=weight, title=edge_type)
        
        # Save and show the graph
        net.save_graph("knowledge_graph.html")
        
        # Read HTML file
        with open("knowledge_graph.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Display in Streamlit
        st.header("Knowledge Graph Visualization")
        html(html_content, height=600)

def display_performance_metrics():
    """Display performance metrics including ROUGE scores"""
    if not st.session_state.performance_metrics:
        st.info("No performance metrics available yet. Add ground truth answers and ask questions to generate metrics.")
        return
    
    # Create dataframe for metrics
    df = pd.DataFrame(st.session_state.performance_metrics)
    
    st.subheader("ROUGE Performance Metrics")
    
    # Summary statistics
    avg_metrics = {
        'ROUGE-1-F': df['rouge-1']['f'].mean(),
        'ROUGE-2-F': df['rouge-2']['f'].mean(),
        'ROUGE-L-F': df['rouge-l']['f'].mean(),
        'ROUGE-1-P': df['rouge-1']['p'].mean(),
        'ROUGE-1-R': df['rouge-1']['r'].mean(),
        'ROUGE-L-P': df['rouge-l']['p'].mean(),
        'ROUGE-L-R': df['rouge-l']['r'].mean(),
        'Avg Response Time (s)': df['time_taken'].mean()
    }
    
    # Display summary metrics
    st.write("Average Performance Metrics:")
    summary_df = pd.DataFrame([avg_metrics])
    st.dataframe(summary_df)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ROUGE-1, ROUGE-2, ROUGE-L F1 scores per query
    axes[0, 0].bar(range(len(df)), df['rouge-1']['f'], label='ROUGE-1-F')
    axes[0, 0].bar(range(len(df)), df['rouge-2']['f'], label='ROUGE-2-F', alpha=0.7)
    axes[0, 0].bar(range(len(df)), df['rouge-l']['f'], label='ROUGE-L-F', alpha=0.5)
    axes[0, 0].set_title('ROUGE F1 Scores by Query')
    axes[0, 0].set_xlabel('Query Index')
    axes[0, 0].set_ylabel('F1 Score')
    axes[0, 0].legend()
    
    # Precision vs Recall for ROUGE-1
    axes[0, 1].scatter(df['rouge-1']['p'], df['rouge-1']['r'], alpha=0.7)
    axes[0, 1].set_title('ROUGE-1: Precision vs Recall')
    axes[0, 1].set_xlabel('Precision')
    axes[0, 1].set_ylabel('Recall')
    
    # Time taken per query
    axes[1, 0].plot(range(len(df)), df['time_taken'], marker='o')
    axes[1, 0].set_title('Response Time by Query')
    axes[1, 0].set_xlabel('Query Index')
    axes[1, 0].set_ylabel('Time (seconds)')
    
    # Average ROUGE F1 scores
    metrics_labels = ['ROUGE-1-F', 'ROUGE-2-F', 'ROUGE-L-F']
    metrics_values = [avg_metrics['ROUGE-1-F'], avg_metrics['ROUGE-2-F'], avg_metrics['ROUGE-L-F']]
    axes[1, 1].bar(metrics_labels, metrics_values)
    axes[1, 1].set_title('Average ROUGE F1 Scores')
    axes[1, 1].set_ylabel('Score')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display individual query metrics
    st.subheader("Individual Query Metrics")
    
    # Format the dataframe for display
    display_df = pd.DataFrame({
        'Query': df['query'],
        'ROUGE-1-F': df['rouge-1']['f'],
        'ROUGE-2-F': df['rouge-2']['f'],
        'ROUGE-L-F': df['rouge-l']['f'],
        'Time (s)': df['time_taken']
    })
    
    st.dataframe(display_df)
    
    # Show detailed comparison
    if st.checkbox("Show Detailed Answer Comparisons"):
        for i, metric in enumerate(st.session_state.performance_metrics):
            with st.expander(f"Query {i+1}: {metric['query'][:50]}..."):
                st.write("Generated Answer:")
                st.write(metric['answer'])
                st.write("Ground Truth Answer:")
                st.write(metric['ground_truth'])
                st.write("ROUGE-1 Scores:")
                st.write(f"F1: {metric['rouge-1']['f']:.4f}, Precision: {metric['rouge-1']['p']:.4f}, Recall: {metric['rouge-1']['r']:.4f}")
                st.write("ROUGE-L Scores:")
                st.write(f"F1: {metric['rouge-l']['f']:.4f}, Precision: {metric['rouge-l']['p']:.4f}, Recall: {metric['rouge-l']['r']:.4f}")

# Initialize button
if uploaded_files and not st.session_state.initialized:
    if st.button("Initialize System"):
        initialize_system(uploaded_files)
elif st.session_state.initialized:
    st.success(f"System initialized with {len(st.session_state.uploaded_files)} PDF files: {', '.join(st.session_state.uploaded_files)}")

# Question answering interface
if st.session_state.initialized:
    tab1, tab2, tab3 = st.tabs(["Question Answering", "Performance Metrics", "Ground Truth Management"])
    
    with tab1:
        st.header("Ask a Question")
        query = st.text_input("Enter your question about the PDF content:")
        
        if query:
            with st.spinner("Searching for answers..."):
                answer = answer_question(query)
                st.subheader("Answer")
                st.write(answer)
                
                # Check if ground truth exists and calculate ROUGE
                if query in st.session_state.ground_truth:
                    ground_truth = st.session_state.ground_truth[query]
                    rouge_scores = calculate_rouge_scores(ground_truth, answer)
                    
                    if rouge_scores:
                        # Get response time from history
                        response_time = next((item['time_taken'] for item in st.session_state.response_history 
                                            if item['query'] == query), 0)
                        
                        # Add to performance metrics
                        st.session_state.performance_metrics.append({
                            'query': query,
                            'answer': answer,
                            'ground_truth': ground_truth,
                            'rouge-1': rouge_scores['rouge-1'],
                            'rouge-2': rouge_scores['rouge-2'],
                            'rouge-l': rouge_scores['rouge-l'],
                            'time_taken': response_time
                        })
                        
                        # Show rouge scores
                        st.info(f"""
                        ROUGE Scores:
                        - ROUGE-1 F1: {rouge_scores['rouge-1']['f']:.4f}
                        - ROUGE-2 F1: {rouge_scores['rouge-2']['f']:.4f}
                        - ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}
                        """)
                
                # Show context sources
                st.subheader("Relevant Context")
                context_docs = enhanced_retrieval(query)
                for i, doc in enumerate(context_docs):
                    with st.expander(f"Context Source {i+1}"):
                        if hasattr(doc, 'page_content'):
                            st.write(doc.page_content)
                        else:
                            st.write(doc)

        # Graph visualization
        if st.button("Show Knowledge Graph"):
            visualize_graph()
    
    with tab2:
        st.header("Performance Analysis")
        if st.button("View Performance Metrics"):
            display_performance_metrics()
    
    with tab3:
        st.header("Ground Truth Management")
        st.write("Add ground truth answers to measure system performance with ROUGE metrics.")
        
        # Add new ground truth
        new_query = st.text_input("Question:", key="gt_question")
        new_ground_truth = st.text_area("Ground Truth Answer:", key="gt_answer")
        
        if st.button("Add Ground Truth"):
            if new_query and new_ground_truth:
                st.session_state.ground_truth[new_query] = new_ground_truth
                st.success(f"Ground truth added for: '{new_query}'")
        
        # View/edit existing ground truths
        st.subheader("Existing Ground Truth Entries")
        if st.session_state.ground_truth:
            for i, (q, a) in enumerate(st.session_state.ground_truth.items()):
                with st.expander(f"Entry {i+1}: {q[:50]}..."):
                    st.write("Question:")
                    st.write(q)
                    st.write("Ground Truth Answer:")
                    st.write(a)
                    
                    # Option to edit
                    edited_answer = st.text_area("Edit Answer:", a, key=f"edit_{i}")
                    if st.button("Update", key=f"update_{i}"):
                        st.session_state.ground_truth[q] = edited_answer
                        st.success("Ground truth updated!")
                    
                    # Option to delete
                    if st.button("Delete", key=f"delete_{i}"):
                        del st.session_state.ground_truth[q]
                        st.success("Ground truth deleted!")
                        st.experimental_rerun()
        else:
            st.info("No ground truth entries yet. Add some to measure performance.")