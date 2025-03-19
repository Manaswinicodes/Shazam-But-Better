import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import seaborn as sns
import os
from tqdm import tqdm

class SearchEngineEvaluator:
    def __init__(self, search_engine):
        """
        Initialize the evaluator with a search engine instance
        
        Parameters:
        -----------
        search_engine : SubtitleSearchEngine or EnhancedSubtitleSearchEngine
            An initialized search engine instance
        """
        self.search_engine = search_engine
        
    def evaluate_queries(self, test_queries, ground_truth, top_n=10):
        """
        Evaluate search engine performance on a set of test queries
        
        Parameters:
        -----------
        test_queries : list
            List of query strings to test
        ground_truth : dict
            Dictionary mapping queries to relevant document IDs
        top_n : int, optional
            Number of top results to consider
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        results = {}
        
        # Metrics to calculate
        precision_at_k = []
        recall_at_k = []
        mean_avg_precision = []
        
        for query in tqdm(test_queries, desc="Evaluating queries"):
            # Get search results
            if hasattr(self.search_engine, 'hybrid_search'):
                search_results = self.search_engine.hybrid_search(query, top_n=top_n)
            else:
                search_results = self.search_engine.search(query, top_n=top_n)
                
            retrieved_docs = [result['file_name'] for result in search_results]
            
            # Get relevant documents for this query
            relevant_docs = ground_truth.get(query, [])
            
            # Calculate precision and recall at k
            relevant_retrieved = set(retrieved_docs).intersection(set(relevant_docs))
            precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0
            recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
            
            precision_at_k.append(precision)
            recall_at
