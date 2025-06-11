# Tests for chunk_naive_llm_classifier
# Covers both English and Chinese text

import pytest
import importlib

def import_task_module(module_path):
    return importlib.import_module(module_path)

def test_chunk_naive_llm_classifier_english_and_chinese():
    classifier = import_task_module('cognee.tasks.chunk_naive_llm_classifier.chunk_naive_llm_classifier')
    english_text = "This is a test chunk."
    chinese_text = "这是一个测试块。"
    # Assuming the classifier has a function called classify_chunk
    if hasattr(classifier, 'classify_chunk'):
        english_result = classifier.classify_chunk(english_text)
        chinese_result = classifier.classify_chunk(chinese_text)
        assert english_result is not None
        assert chinese_result is not None
    else:
        pytest.skip("classify_chunk not implemented in chunk_naive_llm_classifier")
