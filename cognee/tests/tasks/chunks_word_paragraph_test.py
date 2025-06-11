# Tests for chunk_by_word and chunk_by_paragraph
# Covers both English and Chinese text

import pytest
import importlib

def import_task_module(module_path):
    return importlib.import_module(module_path)

def test_chunk_by_word_english_and_chinese():
    chunk_by_word = import_task_module('cognee.tasks.chunks.chunk_by_word')
    english_text = "Hello world! This is a test."
    chinese_text = "你好世界！这是一个测试。"
    english_chunks = chunk_by_word.chunk_by_word(english_text)
    chinese_chunks = chunk_by_word.chunk_by_word(chinese_text)
    assert any("Hello" in c for c in english_chunks)
    assert any("你好" in c for c in chinese_chunks)

def test_chunk_by_paragraph_english_and_chinese():
    chunk_by_paragraph = import_task_module('cognee.tasks.chunks.chunk_by_paragraph')
    english_text = "Paragraph one.\n\nParagraph two."
    chinese_text = "第一段。\n\n第二段。"
    english_chunks = chunk_by_paragraph.chunk_by_paragraph(english_text)
    chinese_chunks = chunk_by_paragraph.chunk_by_paragraph(chinese_text)
    assert any("Paragraph one" in c for c in english_chunks)
    assert any("第一段" in c for c in chinese_chunks)
