import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
from tqdm import tqdm

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SubtitleSearchEngine:
    def __init__(self, data_path, sample_size=None):
        """
        Initialize the subtitle search engine.
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing subtitle data files
        sample_size : float, optional
            Percentage of data to sample (between 0 and 1)
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.subtitles = []
        self.vectorizer = None
        self.subtitle_vectors = None
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load subtitle data from files in the specified directory"""
        print("Loading subtitle data...")
        
        # Get all subtitle files in the directory
        subtitle_files = [f for f in os.listdir(self.data_path) if f.endswith('.srt')]
        
        if self.sample_size:
            # If sample_size is specified, randomly select a subset of files
            num_files = max(1, int(len(subtitle_files) * self.sample_size))
            subtitle_files = random.sample(subtitle_files, num_files)
            print(f"Sampling {num_files} files ({self.sample_size*100:.1f}% of total)")
        
        for file_name in tqdm(subtitle_files, desc="Loading files"):
            file_path = os.path.join(self.data_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Store the subtitle content with its filename
                    self.subtitles.append({
                        'file_name': file_name,
                        'content': content
                    })
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")
        
        print(f"Loaded {len(self.subtitles)} subtitle files")
        return self.subtitles
    
    def preprocess_subtitle(self, subtitle_text):
        """
        Preprocess subtitle text by removing timestamps and cleaning
        
        Parameters:
        -----------
        subtitle_text : str
            Raw subtitle text
        
        Returns:
        --------
        str
            Cleaned subtitle text
        """
        # Remove timestamp patterns (e.g., 00:00:20,000 --> 00:00:24,400)
        cleaned_text = re.sub(r'\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+', '', subtitle_text)
        
        # Remove subtitle numbers
        cleaned_text = re.sub(r'^\d+$', '', cleaned_text, flags=re.MULTILINE)
        
        # Remove HTML tags if any
        cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
    
    def preprocess_all_subtitles(self):
        """Preprocess all loaded subtitles"""
        print("Preprocessing subtitles...")
        for i, subtitle in enumerate(tqdm(self.subtitles, desc="Preprocessing")):
            self.subtitles[i]['cleaned_content'] = self.preprocess_subtitle(subtitle['content'])
        return self.subtitles
    
    def vectorize_documents(self):
        """Vectorize all preprocessed subtitle documents"""
        print("Vectorizing subtitle documents...")
        # Create a TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        
        # Extract the cleaned content from subtitles
        cleaned_contents = [subtitle['cleaned_content'] for subtitle in self.subtitles]
        
        # Fit and transform the documents
        self.subtitle_vectors = self.vectorizer.fit_transform(cleaned_contents)
        
        print(f"Created {self.subtitle_vectors.shape[1]} features for {self.subtitle_vectors.shape[0]} documents")
        return self.subtitle_vectors
    
    def vectorize_query(self, query):
        """
        Vectorize a user query
        
        Parameters:
        -----------
        query : str
            User search query
        
        Returns:
        --------
        scipy.sparse.csr.csr_matrix
            Vectorized query
        """
        # Ensure vectorizer is initialized
        if self.vectorizer is None:
            raise ValueError("Vectorizer not initialized. Call vectorize_documents() first.")
        
        # Transform the query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        return query_vector
    
    def calculate_similarity(self, query_vector):
        """
        Calculate cosine similarity between query and all subtitle documents
        
        Parameters:
        -----------
        query_vector : scipy.sparse.csr.csr_matrix
            Vectorized query
        
        Returns:
        --------
        numpy.ndarray
            Array of similarity scores
        """
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.subtitle_vectors).flatten()
        return similarities
    
    def search(self, query, top_n=5):
        """
        Search for subtitles matching the query
        
        Parameters:
        -----------
        query : str
            User search query
        top_n : int, optional
            Number of top results to return
        
        Returns:
        --------
        list
            List of top matching subtitles with similarity scores
        """
        # Vectorize the query
        query_vector = self.vectorize_query(query)
        
        # Calculate similarities
        similarities = self.calculate_similarity(query_vector)
        
        # Get indices of top results
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            results.append({
                'file_name': self.subtitles[idx]['file_name'],
                'similarity': similarities[idx],
                'content_preview': self.subtitles[idx]['cleaned_content'][:150] + '...' if len(self.subtitles[idx]['cleaned_content']) > 150 else self.subtitles[idx]['cleaned_content']
            })
        
        return results

    def save_model(self, filepath):
        """
        Save the search engine model to a file
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import pickle
        
        model_data = {
            'vectorizer': self.vectorizer,
            'subtitle_vectors': self.subtitle_vectors,
            'subtitles': self.subtitles
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a search engine model from a file
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        SubtitleSearchEngine
            Loaded search engine instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        search_engine = cls(data_path=None)
        
        # Restore the model data
        search_engine.vectorizer = model_data['vectorizer']
        search_engine.subtitle_vectors = model_data['subtitle_vectors']
        search_engine.subtitles = model_data['subtitles']
        
        print(f"Model loaded from {filepath}")
        return search_engine

# Demo usage
def main():
    # Path to the directory containing subtitle files
    data_path = "../data"  # Adjust as needed
    
    # Initialize the search engine with 30% of the data
    search_engine = SubtitleSearchEngine(data_path, sample_size=0.3)
    
    # Load and preprocess the data
    search_engine.load_data()
    search_engine.preprocess_all_subtitles()
    search_engine.vectorize_documents()
    
    # Save the model
    search_engine.save_model("../models/search_engine_model.pkl")
    
    # Example search queries
    queries = [
        "global warming",
        "artificial intelligence",
        "quantum physics explanation"
    ]
    
    # Perform searches
    for query in queries:
        print(f"\nSearch results for: '{query}'")
        results = search_engine.search(query, top_n=3)
        
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['file_name']} (Similarity: {result['similarity']:.4f})")
            print(f"Preview: {result['content_preview']}")

if __name__ == "__main__":
    main()
