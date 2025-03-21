import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path so we can import the project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.search_engine import SubtitleIndex, SearchEngine
from src.evaluation import evaluate_search_results

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple test subtitle file
        self.test_srt_path = os.path.join(self.test_dir, "test.srt")
        with open(self.test_srt_path, "w") as f:
            f.write("""1
00:00:01,000 --> 00:00:04,000
Hello world, this is a test subtitle.

2
00:00:05,000 --> 00:00:08,000
We are testing the search engine functionality.

3
00:00:09,000 --> 00:00:12,000
The search engine should find this sentence.

4
00:00:13,000 --> 00:00:16,000
Multiple words like search and engine should be found.
""")
        
        # Initialize the search engine with the test file
        self.index = SubtitleIndex()
        self.index.add_subtitle_file(self.test_srt_path, "test")
        self.engine = SearchEngine(self.index)
    
    def tearDown(self):
        # Clean up temporary directory after tests
        shutil.rmtree(self.test_dir)
    
    def test_basic_search(self):
        # Test exact match search
        results = self.engine.search("test subtitle")
        self.assertTrue(len(results) > 0)
        self.assertIn("test subtitle", results[0]["text"].lower())
    
    def test_partial_search(self):
        # Test partial match search
        results = self.engine.search("search engine")
        self.assertTrue(len(results) >= 2)  # Should find at least two matches
        
        # Check that results contain the search terms
        found_sentence = False
        found_multiple = False
        for result in results:
            if "should find this sentence" in result["text"].lower():
                found_sentence = True
            if "multiple words" in result["text"].lower():
                found_multiple = True
        
        self.assertTrue(found_sentence)
        self.assertTrue(found_multiple)
    
    def test_no_results(self):
        # Test search with no matches
        results = self.engine.search("nonexistent phrase xyz")
        self.assertEqual(len(results), 0)
    
    def test_case_insensitivity(self):
        # Test case insensitive search
        results_lower = self.engine.search("hello world")
        results_upper = self.engine.search("HELLO WORLD")
        
        self.assertEqual(len(results_lower), len(results_upper))
        self.assertEqual(results_lower[0]["text"], results_upper[0]["text"])
    
    def test_ranking(self):
        # Test that results are properly ranked
        results = self.engine.search("search engine")
        
        # Check if results with both terms are ranked higher
        # than results with only one term
        if len(results) >= 2:
            text1 = results[0]["text"].lower()
            text2 = results[1]["text"].lower()
            
            # Count occurrences of search terms in each result
            count1 = text1.count("search") + text1.count("engine")
            count2 = text2.count("search") + text2.count("engine")
            
            # The first result should have more occurrences or be a better match
            self.assertTrue(count1 >= count2)
    
    def test_metadata_in_results(self):
        # Test that results include correct metadata
        results = self.engine.search("test")
        self.assertTrue(len(results) > 0)
        
        # Check metadata fields
        result = results[0]
        self.assertIn("file_id", result)
        self.assertIn("text", result)
        self.assertIn("start_time", result)
        self.assertIn("end_time", result)
        self.assertIn("score", result)
        
        # Check file_id is correct
        self.assertEqual(result["file_id"], "test")

if __name__ == "__main__":
    unittest.main()
