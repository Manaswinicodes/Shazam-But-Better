import unittest
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import evaluate_search_results, compute_metrics

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        # Mock search results for testing
        self.search_results = [
            {"file_id": "movie1", "text": "This is the first result", "start_time": "00:01:00", "end_time": "00:01:05", "score": 0.95},
            {"file_id": "movie1", "text": "This is the second result", "start_time": "00:02:00", "end_time": "00:02:05", "score": 0.85},
            {"file_id": "movie2", "text": "This is the third result", "start_time": "00:03:00", "end_time": "00:03:05", "score": 0.75},
            {"file_id": "movie3", "text": "This is the fourth result", "start_time": "00:04:00", "end_time": "00:04:05", "score": 0.65},
            {"file_id": "movie3", "text": "This is the fifth result", "start_time": "00:05:00", "end_time": "00:05:05", "score": 0.55},
        ]
        
        # Mock ground truth for testing
        self.ground_truth = [
            {"file_id": "movie1", "text": "This is the first result", "start_time": "00:01:00", "end_time": "00:01:05"},
            {"file_id": "movie2", "text": "This is the third result", "start_time": "00:03:00", "end_time": "00:03:05"},
            {"file_id": "movie4", "text": "This is a missing result", "start_time": "00:06:00", "end_time": "00:06:05"},
        ]
    
    def test_evaluation_metrics(self):
        # Test the compute_metrics function
        metrics = compute_metrics(self.search_results[:3], self.ground_truth)
        
        # Check that all expected metrics are returned
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1_score", metrics)
        self.assertIn("average_precision", metrics)
        
        # Check metric values
        # In our mock data, 2 out of 3 results are relevant
        self.assertAlmostEqual(metrics["precision"], 2/3, places=2)
        # 2 out of 3 ground truth items are found
        self.assertAlmostEqual(metrics["recall"], 2/3, places=2)
        # F1 is harmonic mean of precision and recall
        self.assertAlmostEqual(metrics["f1_score"], 2/3, places=2)
    
    def test_precision_at_k(self):
        # Test precision at different k values
        metrics_k1 = compute_metrics(self.search_results[:1], self.ground_truth)
        metrics_k3 = compute_metrics(self.search_results[:3], self.ground_truth)
        metrics_k5 = compute_metrics(self.search_results[:5], self.ground_truth)
        
        # Precision@1 should be 1.0 (the top result is relevant)
        self.assertAlmostEqual(metrics_k1["precision"], 1.0, places=2)
        
        # Precision@3 should be 2/3 (2 out of 3 are relevant)
        self.assertAlmostEqual(metrics_k3["precision"], 2/3, places=2)
        
        # Precision@5 should be 2/5 (2 out of 5 are relevant)
        self.assertAlmostEqual(metrics_k5["precision"], 2/5, places=2)
    
    def test_recall_at_k(self):
        # Test recall at different k values
        metrics_k1 = compute_metrics(self.search_results[:1], self.ground_truth)
        metrics_k3 = compute_metrics(self.search_results[:3], self.ground_truth)
        metrics_k5 = compute_metrics(self.search_results[:5], self.ground_truth)
        
        # Recall@1 should be 1/3 (1 out of 3 ground truth items)
        self.assertAlmostEqual(metrics_k1["recall"], 1/3, places=2)
        
        # Recall@3 should be 2/3 (2 out of 3 ground truth items)
        self.assertAlmostEqual(metrics_k3["recall"], 2/3, places=2)
        
        # Recall@5 should be 2/3 (still only 2 out of 3 ground truth items)
        self.assertAlmostEqual(metrics_k5["recall"], 2/3, places=2)
    
    def test_empty_results(self):
        # Test with empty results
        metrics = compute_metrics([], self.ground_truth)
        
        # Precision is undefined but set to 0
        self.assertEqual(metrics["precision"], 0)
        # Recall should be 0
        self.assertEqual(metrics["recall"], 0)
        # F1 should be 0
        self.assertEqual(metrics["f1_score"], 0)
    
    def test_no_relevant_results(self):
        # Test with no relevant results
        irrelevant_results = [
            {"file_id": "movie5", "text": "This is irrelevant", "start_time": "00:10:00", "end_time": "00:10:05", "score": 0.9},
            {"file_id": "movie6", "text": "Also irrelevant", "start_time": "00:11:00", "end_time": "00:11:05", "score": 0.8},
        ]
        
        metrics = compute_metrics(irrelevant_results, self.ground_truth)
        
        # Precision should be 0
        self.assertEqual(metrics["precision"], 0)
        # Recall should be 0
        self.assertEqual(metrics["recall"], 0)
        # F1 should be 0
        self.assertEqual(metrics["f1_score"], 0)
    
    def test_full_evaluation_function(self):
        # Test the main evaluation function
        evaluation_results = evaluate_search_results(
            self.search_results, 
            self.ground_truth,
            k_values=[1, 3, 5]
        )
        
        # Check that results for each k value are included
        self.assertIn("overall", evaluation_results)
        self.assertIn("precision_at_k", evaluation_results)
        self.assertIn("recall_at_k", evaluation_results)
        
        # Check that each k value is included
        self.assertIn(1, evaluation_results["precision_at_k"])
        self.assertIn(3, evaluation_results["precision_at_k"])
        self.assertIn(5, evaluation_results["precision_at_k"])
        
        # Check overall metrics
        overall = evaluation_results["overall"]
        self.assertIn("precision", overall)
        self.assertIn("recall", overall)
        self.assertIn("f1_score", overall)

if __name__ == "__main__":
    unittest.main()
