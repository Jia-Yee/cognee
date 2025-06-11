# Tests for extract_chunks_from_documents and related document tasks
# Covers both English and Chinese text

import pytest
import importlib
import sys
import asyncio
import types

def import_task_module(module_path):
    return importlib.import_module(module_path)

async def collect_async_generator(async_gen):
    result = []
    async for item in async_gen:
        result.append(item)
    return result

def test_extract_chunks_from_documents_english_and_chinese():
    extract_chunks = import_task_module('cognee.tasks.documents.extract_chunks_from_documents')
    english_doc = {'document_id': 'eng1', 'text': 'This is a test. Another sentence.'}
    chinese_doc = {'document_id': 'zh1', 'text': '这是一个测试。另一个句子。'}
    english_chunks_result = extract_chunks.extract_chunks_from_documents([english_doc], 'sentence')
    chinese_chunks_result = extract_chunks.extract_chunks_from_documents([chinese_doc], 'sentence')
    if isinstance(english_chunks_result, types.AsyncGeneratorType):
        english_chunks = asyncio.get_event_loop().run_until_complete(collect_async_generator(english_chunks_result))
    else:
        english_chunks = list(english_chunks_result)
    if isinstance(chinese_chunks_result, types.AsyncGeneratorType):
        chinese_chunks = asyncio.get_event_loop().run_until_complete(collect_async_generator(chinese_chunks_result))
    else:
        chinese_chunks = list(chinese_chunks_result)
    assert any('test' in c['text'] for c in english_chunks)
    assert any('测试' in c['text'] for c in chinese_chunks)

# More tests for other document tasks can be added here.
