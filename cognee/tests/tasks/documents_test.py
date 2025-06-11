# Tests for extract_chunks_from_documents and related document tasks
# Covers both English and Chinese text

import pytest
import importlib

def import_task_module(module_path):
    return importlib.import_module(module_path)

def test_extract_chunks_from_documents_english_and_chinese():
    extract_chunks = import_task_module('cognee.tasks.documents.extract_chunks_from_documents')
    # Simulate a document input structure as expected by the function
    english_doc = {'text': 'This is a test. Another sentence.'}
    chinese_doc = {'text': '这是一个测试。另一个句子。'}
    # The function may require a list of documents
    english_chunks = extract_chunks.extract_chunks_from_documents([english_doc], chunker_name='sentence')
    chinese_chunks = extract_chunks.extract_chunks_from_documents([chinese_doc], chunker_name='sentence')
    assert any('test' in c['text'] for c in english_chunks)
    assert any('测试' in c['text'] for c in chinese_chunks)

# More tests for other document tasks can be added here.
