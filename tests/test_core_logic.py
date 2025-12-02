import unittest
import sys
import os
import numpy as np
from unittest.mock import MagicMock

# Mock sentence_transformers and streamlit before importing core_logic
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['streamlit'] = MagicMock()
sys.modules['dashscope'] = MagicMock()
sys.modules['jieba'] = MagicMock()

# Setup jieba mock behavior
sys.modules['jieba'].cut.side_effect = lambda x: x.split() # Simple space split for testing if needed, or just list of chars if we want. 
# But wait, the test input "假如张居正改革" has no spaces.
# Let's make the mock smarter or change the test input to be space separated for the mock to work easily?
# Or just return a fixed list for the specific test input.

def jieba_cut_mock(text):
    if text == "假如张居正改革":
        return ["假如", "张居正", "改革"]
    return list(text)

sys.modules['jieba'].cut.side_effect = jieba_cut_mock

# Add parent directory to path to import core_logic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_logic import ContextAlignmentLayer, FictionDiffusionLayer, ContentAuditor

class MockEmbeddingLayer:
    def search(self, vec, top_k=3):
        # Return dummy results
        return [
            {"data": {"id": "1", "text": "test1", "name": "test1"}, "score": 0.9},
            {"data": {"id": "2", "text": "test2", "name": "test2"}, "score": 0.8}
        ]

class TestCoreLogic(unittest.TestCase):

    def test_context_alignment_layer(self):
        layer = ContextAlignmentLayer()
        # Test with keywords
        result = layer.validate("这是一个关于锦衣卫和内阁的故事")
        self.assertTrue(result['is_valid'])
        self.assertIn("锦衣卫", result['keywords'])
        self.assertIn("内阁", result['keywords'])
        
        # Test without keywords
        result = layer.validate("这是一个普通的故事")
        self.assertFalse(result['is_valid'])

    def test_fiction_diffusion_layer(self):
        mock_emb = MockEmbeddingLayer()
        layer = FictionDiffusionLayer(mock_emb)
        
        fact_vec = np.array([1.0, 0.0])
        query_vec = np.array([0.0, 1.0])
        
        # Test interpolation
        gen_vec, results = layer.interpolate_and_generate(fact_vec, query_vec, alpha=0.5)
        
        # Expected vector is normalized [0.5, 0.5] -> [0.707, 0.707]
        expected = np.array([0.5, 0.5])
        expected = expected / np.linalg.norm(expected)
        
        np.testing.assert_array_almost_equal(gen_vec, expected)
        self.assertEqual(len(results), 2)

    def test_content_auditor(self):
        auditor = ContentAuditor()
        
        # Test pass
        result = auditor.audit("假如张居正改革", "张居正进行了改革")
        self.assertTrue(result['passed'])
        
        # Test fail (missing entity)
        result = auditor.audit("假如张居正改革", "李四进行了改革")
        self.assertFalse(result['passed'])
        self.assertIn("张居正", result['message'])

if __name__ == '__main__':
    unittest.main()
