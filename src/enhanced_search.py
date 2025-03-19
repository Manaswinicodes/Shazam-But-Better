import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import pickle
from tqdm import tqdm

class EnhancedSubtitleSearchEngine:
    def __init__(self, base_search_engine=None, embedding_model=None):
        """
        Initialize enhanced search engine with advanced features
        
        Parameters:
        -----------
        base_search_engine : SubtitleSearchEngine, optional
            Base search engine to enhance
        embedding_model : str, optional
            Name of the sentence transformer model to use
        """
        self.base_engine = base_search_engine
        self.embedding_model = embedding_model if embedding_model else 'all-MiniLM-L6-v2'
        self.transformer = None
        self.subtitle_embeddings = None
        self.subtitle_clusters = None
        self.vectorizer = None
        self.subtitle_vectors = None
        self.subtitles = []
        
    def load_base_engine_data(self):
        """Load data from the base search engine"""
        if self.base_engine is None:
            raise ValueError("Base search engine not provided")
        
        self.subtitles = self.base_engine.subtitles
        self.vectorizer = self.base_engine.vectorizer
        self.subtitle_vectors = self.base_engine.subtitle_vectors
        print(f"Loaded data from base engine: {len(self.subtitles)} subtitles")
        
    def initialize_transformer(self):
        """Initialize the sentence transformer model"""
        print(f"Initializing sentence transformer model: {self.embedding_model}")
        try:
            from sentence_transformers import SentenceTransformer
            self.transformer = SentenceTransformer(self.embedding_model)
            print("Transformer model initialized successfully")
        except Exception as e:
            print(f"Error initializing transformer model: {e}")
            print("Falling back to TF-IDF vectorization")
    
    def create_semantic_embeddings(self):
        """Create semantic embeddings for subtitle content using transformer"""
        if self.transformer is None:
            self.initialize_transformer()
        
        print("Creating semantic embeddings...")
        subtitle_texts = [subtitle['cleaned_content'] for subtitle in self.subtitles]
        
        # Split long texts into chunks for better processing
        chunked_texts = []
        chunk_map = []  # Map to track which chunks belong to which document
        
        for i, text in enumerate(tqdm(subtitle_texts, desc="Chunking texts")):
            # Split text into chunks of approximately 500 characters
            chunks = [text[j:j+500] for j in range(0, len(text), 500)]
            chunked_texts.extend(chunks)
            chunk_map.extend([i] * len(chunks))
        
        # Create embeddings for chunks
        print(f"Encoding {len(chunked_texts)} text chunks...")
        chunk_embeddings = self.transformer.encode(chunked_texts, show_progress_bar=True)
        
        # Aggregate chunk embeddings back to documents
        print("Aggregating embeddings...")
        doc_embeddings = np.zeros((len(subtitle_texts), chunk_embeddings.shape[1]))
        for i, chunk_idx in enumerate(tqdm(chunk_map, desc="Aggregating")):
            doc_embeddings[chunk_idx] += chunk_embeddings[i]
        
        # Normalize to account for different numbers of chunks
        for i in range(len(subtitle_texts)):
            doc_count = chunk_map.count(i)
            if doc_count > 0:
                doc_embeddings[i] /= doc_count
        
        # Normalize embeddings
        self.subtitle_embeddings = normalize(doc_embeddings)
        print(f"Created semantic embeddings with shape: {self.subtitle_embeddings.shape}")
        
        return self.subtitle_embeddings
    
    def cluster_subtitles(self, n_clusters=10):
        """
        Cluster subtitles based on semantic similarity
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters to create
        """
        if self.subtitle_embeddings is None:
            self.create_semantic_embeddings()
        
        print(f"Clustering subtitles into {n_clusters} groups...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.subtitle_clusters = kmeans.fit_predict(self.subtitle_embeddings)
        
        # Add cluster information to subtitles
        for i, subtitle in enumerate(self.subtitles):
            subtitle['cluster'] = int(self.subtitle_clusters[i])
        
        # Analyze clusters
        cluster_counts = np.bincount(self.subtitle_clusters)
        print("Cluster distribution:")
        for cluster_id, count in enumerate(cluster_counts):
            print(f"  Cluster {cluster_id}: {count} subtitles")
        
        return self.subtitle_clusters
    
    def get_cluster_keywords(self, cluster_id, top_n=10):
        """
        Get top keywords for a specific cluster
        
        Parameters:
        -----------
        cluster_id : int
            ID of the cluster
        top_n : int, optional
            Number of top keywords to return
        
        Returns:
        --------
        list
            List of top keywords for the cluster
        """
        if self.subtitle_clusters is None:
            self.cluster_subtitles()
        
        # Get subtitles in this cluster
        cluster_indices = np.where(self.subtitle_clusters == cluster_id)[0]
        cluster_subtitles = [self.subtitles[i]['cleaned_content'] for i in cluster_indices]
        
        # Create a new vectorizer for this cluster
        from sklearn.feature_extraction.text import TfidfVectorizer
        cluster_vectorizer = TfidfVectorizer(max_features=100)
        cluster_vectors = cluster_vectorizer.fit_transform(cluster_subtitles)
        
        # Get top feature names
        feature_names = cluster_vectorizer.get_feature_names_out()
        tfidf_sums = cluster_vectors.sum(axis=0).A1
        top_indices = tfidf_sums.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        
        return top_keywords
    
    def visualize_clusters(self, output_path=None):
        """
        Visualize subtitle clusters using dimensionality reduction
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the visualization images
        """
        if self.subtitle_embeddings is None:
            self.create_semantic_embeddings()
        
        # Reduce dimensionality for visualization
        svd = TruncatedSVD(n_components=2)
        reduced_embeddings = svd.fit_transform(self.subtitle_embeddings)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                             c=self.subtitle_clusters, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, label='Cluster')
        plt.title('Subtitle Clusters Visualization')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'cluster_visualization.png'))
            print(f"Saved cluster visualization to {os.path.join(output_path, 'cluster_visualization.png')}")
        else:
            plt.show()
        
        # Generate word clouds for top clusters
        top_clusters = np.bincount(self.subtitle_clusters).argsort()[-3:][::-1]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, cluster_id in enumerate(top_clusters):
            cluster_indices = np.where(self.subtitle_clusters == cluster_id)[0]
            cluster_text = " ".join([self.subtitles[j]['cleaned_content'] for j in cluster_indices])
            
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'Cluster {cluster_id}: {len(cluster_indices)} subtitles')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'cluster_wordclouds.png'))
            print(f"Saved cluster wordclouds to {os.path.join(output_path, 'cluster_wordclouds.png')}")
        else:
            plt.show()
    
    def hybrid_search(self, query, top_n=5, semantic_weight=0.7):
        """
        Perform hybrid search combining TF-IDF and semantic similarity
        
        Parameters:
        -----------
        query : str
            User search query
        top_n : int, optional
            Number of top results to return
        semantic_weight : float, optional
            Weight for semantic similarity (0 to 1)
        
        Returns:
        --------
        list
            List of top matching subtitles with similarity scores
        """
        if self.subtitle_embeddings is None:
            self.create_semantic_embeddings()
        
        # TF-IDF similarity
        tfidf_query_vector = self.base_engine.vectorize_query(query)
        tfidf_similarities = self.base_engine.calculate_similarity(tfidf_query_vector)
        
        # Semantic similarity
        query_embedding = self.transformer.encode([query])
        semantic_similarities = np.dot(self.subtitle_embeddings, query_embedding.T).flatten()
        
        # Combine similarities
        combined_similarities = (1 - semantic_weight) * tfidf_similarities + semantic_weight * semantic_similarities
        
        # Get indices of top results
        top_indices = combined_similarities.argsort()[-top_n:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            subtitle = self.subtitles[idx]
            results.append({
                'file_name': subtitle['file_name'],
                'similarity': combined_similarities[idx],
                'tfidf_similarity': tfidf_similarities[idx],
                'semantic_similarity': semantic_similarities[idx],
                'cluster': subtitle.get('cluster', -1),
                'content_preview': subtitle['cleaned_content'][:150] + '...' if len(subtitle['cleaned_content']) > 150 else subtitle['cleaned_content']
            })
        
        return results
    
    def save_model(self, filepath):
        """
        Save the enhanced search engine model to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'subtitles': self.subtitles,
            'subtitle_embeddings': self.subtitle_embeddings,
            'subtitle_clusters': self.subtitle_clusters,
            'embedding_model': self.embedding_model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Enhanced model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, base_engine=None):
        """
        Load an enhanced search engine model from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        base_engine : SubtitleSearchEngine, optional
            Base search engine instance
            
        Returns:
        --------
        EnhancedSubtitleSearchEngine
            Loaded enhanced search engine instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        embedding_model = model_data.get('embedding_model', 'all-MiniLM-L6-v2')
        search_engine = cls(base_engine=base_engine, embedding_model=embedding_model)
        
        # Restore the model data
        search_engine.subtitles = model_data['subtitles']
        search_engine.subtitle_embeddings = model_data['subtitle_embeddings']
        search_engine.subtitle_clusters = model_data['subtitle_clusters']
        
        # Initialize transformer
        search_engine.initialize_transformer()
        
        print(f"Enhanced model loaded from {filepath}")
        return search_engine

def main():
    # This is a demo of how to use the enhanced search engine
    from search_engine import SubtitleSearchEngine
    
    # Path to the directory containing subtitle files
    data_path = "../data"  # Adjust as needed
    
    # Initialize the base search engine
    base_engine = SubtitleSearchEngine(data_path, sample_size=0.2)
    
    # Load and preprocess the data
    base_engine.load_data()
    base_engine.preprocess_all_subtitles()
    base_engine.vectorize_documents()
    
    # Initialize the enhanced search engine
    enhanced_engine = EnhancedSubtitleSearchEngine(base_engine=base_engine)
    
    # Load data from the base engine
    enhanced_engine.load_base_engine_data()
    
    # Create semantic embeddings
    enhanced_engine.create_semantic_embeddings()
    
    # Cluster subtitles
    enhanced_engine.cluster_subtitles(n_clusters=8)
    
    # Save the model
    enhanced_engine.save_model("../models/enhanced_search_engine_model.pkl")
    
    # Example search queries
    queries = [
        "global warming",
        "artificial intelligence",
        "quantum physics explanation"
    ]
    
    # Perform searches
    for query in queries:
        print(f"\nSearch results for: '{query}'")
        results = enhanced_engine.hybrid_search(query, top_n=3)
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['file_name']} (Similarity: {result['similarity']:.4f})")
            print(f"TF-IDF: {result['tfidf_similarity']:.4f}, Semantic: {result['semantic_similarity']:.4f}")
            print(f"Cluster: {result['cluster']}")
            print(f"Preview: {result['content_preview']}")

if __name__ == "__main__":
    main()
