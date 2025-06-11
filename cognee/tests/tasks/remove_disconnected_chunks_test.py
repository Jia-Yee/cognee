# Tests for remove_disconnected_chunks
# Covers both English and Chinese text

import pytest
import importlib

def import_task_module(module_path):
    return importlib.import_module(module_path)

def test_remove_disconnected_chunks_english_and_chinese():
    remove_disconnected_chunks = import_task_module('cognee.tasks.chunks.remove_disconnected_chunks')
    # Simulate input: list of chunk dicts with 'connected' field
    english_chunks = [
        {'text': 'Connected chunk.', 'connected': True},
        {'text': 'Disconnected chunk.', 'connected': False}
    ]
    chinese_chunks = [
        {'text': '已连接的块。', 'connected': True},
        {'text': '未连接的块。', 'connected': False}
    ]
    filtered_english = remove_disconnected_chunks.remove_disconnected_chunks(english_chunks)
    filtered_chinese = remove_disconnected_chunks.remove_disconnected_chunks(chinese_chunks)
    assert all(c['connected'] for c in filtered_english)
    assert all(c['connected'] for c in filtered_chinese)
