import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
from pathlib import Path

# Add the src directory to the path so we can import the search engine modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import search engine modules
try:
    from search_engine import SubtitleSearchEngine
    from enhanced_search import EnhancedSubtitleSearchEngine
except ImportError:
    st.error("Could not import search engine modules. Make sure the application structure is correct.")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Subtitle Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Sidebar configuration
st.sidebar.title("Search Configuration")

@st.cache_resource
def load_search_engine(model_path=None, data_path=None, sample_size=0.3, use_enhanced=False):
    """Load or create search engine instances"""
    
    if model_path and os.path.exists(model_path):
        try:
            st.sidebar.success(f"Loading model from {model_path}")
            if use_enhanced:
                # First load base model if it exists
                base_model_path = model_path.replace("enhanced_", "")
                if os.path.exists(base_model_path):
                    base_engine = SubtitleSearchEngine.load_model(base_model_path)
                    return EnhancedSubtitleSearchEngine.load_model(model_path, base_engine=base_engine)
                else:
                    return EnhancedSubtitleSearchEngine.load_model(model_path)
            else:
                return SubtitleSearchEngine.load_model(model_path)
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
    
    if not data_path:
        st.sidebar.error("No data path provided and no model found.")
        st.stop()
        
    # Create new search engine instance
    st.sidebar.info(f"Creating new search engine with {sample_size*100}% of data from {data_path}")
    
    with st.spinner("Initializing search engine..."):
        # Initialize base search engine
        base_engine = SubtitleSearchEngine(data_path, sample_size=sample_size)
        
        # Load and preprocess subtitles
        base_engine.load_data()
        base_engine.preprocess_all_subtitles()
        base_engine.vectorize_documents()
        
        if use_enhanced:
            # Initialize enhanced search engine
            enhanced_engine = EnhancedSubtitleSearchEngine(base_engine=base_engine)
            enhanced_engine.load_base_engine_data()
            enhanced_engine.create_semantic_embeddings()
            enhanced_engine.cluster_subtitles(n_clusters=8)
            return enhanced_engine
        else:
            return base_engine

def save_current_engine(engine, model_dir="models", use_enhanced=False):
    """Save the current search engine model"""
    os.makedirs(model_dir, exist_ok=True)
    
    if use_enhanced:
        model_path = os.path.join(model_dir, "enhanced_search_engine_model.pkl")
    else:
        model_path = os.path.join(model_dir, "search_engine_model.pkl")
    
    engine.save_model(model_path)
    st.sidebar.success(f"Model saved to {model_path}")
    return model_path

# Setup options in sidebar
data_path = st.sidebar.text_input("Data Directory Path", value="data")
model_dir = st.sidebar.text_input("Model Directory Path", value="models")
sample_size = st.sidebar.slider("Sample Size", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
use_enhanced = st.sidebar.checkbox("Use Enhanced Search Engine", value=True)

# Determine model path
if use_enhanced:
    model_path = os.path.join(model_dir, "enhanced_search_engine_model.pkl")
else:
    model_path = os.path.join(model_dir, "search_engine_model.pkl")

# Check if we should create a new model or load existing
create_new = st.sidebar.checkbox("Create new model", value=not os.path.exists(model_path))

if create_new:
    if st.sidebar.button("Initialize Search Engine"):
        search_engine = load_search_engine(
            data_path=data_path, 
            sample_size=sample_size, 
            use_enhanced=use_enhanced
        )
        save_current_engine(search_engine, model_dir, use_enhanced)
else:
    search_engine = load_search_engine(
        model_path=model_path, 
        data_path=data_path, 
        use_enhanced=use_enhanced
    )

# Main application area
st.title("🔍 Subtitle Search Engine")

# Search interface
query = st.text_input("Enter your search query", "")
col1, col2 = st.columns(2)
top_n = col1.number_input("Number of results", min_value=1, max_value=20, value=5)

if use_enhanced and 'search_engine' in locals() and hasattr(search_engine, 'hybrid_search'):
    semantic_weight = col2.slider(
        "Semantic Weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        help="Higher values give more importance to semantic similarity"
    )

# Search button
search_clicked = st.button("Search")

if 'search_engine' in locals() and search_clicked and query:
    with st.spinner("Searching..."):
        if use_enhanced and hasattr(search_engine, 'hybrid_search'):
            results = search_engine.hybrid_search(query, top_n=top_n, semantic_weight=semantic_weight)
            st.success(f"Found {len(results)} results using hybrid search")
        else:
            results = search_engine.search(query, top_n=top_n)
            st.success(f"Found {len(results)} results using TF-IDF search")
    
    # Display results
    for i, result in enumerate(results):
        with st.expander(f"{i+1}. {result['file_name']} (Similarity: {result['similarity']:.4f})", expanded=(i==0)):
            if use_enhanced and 'tfidf_similarity' in result:
                cols = st.columns(3)
                cols[0].metric("Combined Score", f"{result['similarity']:.4f}")
                cols[1].metric("TF-IDF Score", f"{result['tfidf_similarity']:.4f}")
                cols[2].metric("Semantic Score", f"{result['semantic_similarity']:.4f}")
                if 'cluster' in result:
                    st.info(f"Document Cluster: {result['cluster']}")
            
            st.text_area("Content Preview", result['content_preview'], height=150)

# Advanced features (visible only when using enhanced search)
if 'search_engine' in locals() and use_enhanced and hasattr(search_engine, 'subtitle_clusters'):
    st.header("Advanced Analysis")
    
    tab1, tab2 = st.tabs(["Cluster Analysis", "Search Statistics"])
    
    with tab1:
        st.subheader("Subtitle Clusters")
        
        # Select cluster
        n_clusters = len(np.unique(search_engine.subtitle_clusters))
        selected_cluster = st.selectbox(
            "Select Cluster", 
            options=list(range(n_clusters)),
            format_func=lambda x: f"Cluster {x} ({np.sum(search_engine.subtitle_clusters == x)} documents)"
        )
        
        # Get cluster keywords
        if st.button("Analyze Cluster Keywords"):
            with st.spinner("Analyzing cluster keywords..."):
                keywords = search_engine.get_cluster_keywords(selected_cluster, top_n=15)
                
                # Display keywords
                st.write("Top Keywords in Cluster:")
                keyword_df = pd.DataFrame({
                    'Keyword': keywords,
                    'Importance': range(len(keywords), 0, -1)
                })
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(data=keyword_df, x='Importance', y='Keyword', ax=ax)
                ax.set_title(f"Top Keywords for Cluster {selected_cluster}")
                st.pyplot(fig)
                
                # Sample documents from this cluster
                cluster_docs = [
                    doc for i, doc in enumerate(search_engine.subtitles) 
                    if search_engine.subtitle_clusters[i] == selected_cluster
                ]
                
                st.write(f"Sample Documents from Cluster {selected_cluster}:")
                for i, doc in enumerate(cluster_docs[:3]):
                    with st.expander(f"Document {i+1}: {doc['file_name']}"):
                        st.write(doc['cleaned_content'][:500] + "..." if len(doc['cleaned_content']) > 500 else doc['cleaned_content'])
    
    with tab2:
        st.subheader("Search Engine Statistics")
        
        # Basic stats
        st.write(f"Total documents indexed: {len(search_engine.subtitles)}")
        st.write(f"Number of clusters: {n_clusters}")
        
        # Cluster distribution
        cluster_counts = np.bincount(search_engine.subtitle_clusters)
        cluster_df = pd.DataFrame({
            'Cluster': range(len(cluster_counts)),
            'Count': cluster_counts
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=cluster_df, x='Cluster', y='Count', ax=ax)
        ax.set_title("Documents per Cluster")
        st.pyplot(fig)

# About section
with st.sidebar.expander("About"):
    st.write("""
    ## Subtitle Search Engine
    
    This application allows you to search through video subtitle content using advanced 
    natural language processing techniques.
    
    ### Features:
    - Basic TF-IDF search
    - Enhanced semantic search using transformer models
    - Document clustering and analysis
    - Hybrid search combining multiple similarity metrics
    
    ### Usage:
    1. Set up your data directory with .srt subtitle files
    2. Initialize the search engine
    3. Enter search queries to find relevant content
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Subtitle Search Engine v1.0")
