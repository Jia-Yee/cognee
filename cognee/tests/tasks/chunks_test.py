# Tests for chunking and document tasks in cognee/tasks
# Covers both English and Chinese text

import pytest
import importlib

# Helper for dynamic import and test discovery
def import_task_module(module_path):
    return importlib.import_module(module_path)

# Example test for chunk_by_sentence (to be expanded for all modules)
def test_chunk_by_sentence_english_and_chinese():
    chunk_by_sentence = import_task_module('cognee.tasks.chunks.chunk_by_sentence')
    # English
    english_text = "This is a sentence. Here is another one! And a third?"
    english_chunks = list(chunk_by_sentence.chunk_by_sentence(english_text))
    assert any("sentence" in c[1] for c in english_chunks)
    # Chinese
    chinese_text = "这是一个句子。这里还有另一个！再来一个？"
    chinese_chunks = list(chunk_by_sentence.chunk_by_sentence(chinese_text))
    assert any("句子" in c[1] for c in chinese_chunks)

# More tests for chunk_by_word, chunk_by_paragraph, extract_chunks_from_documents, etc. will be added here.
