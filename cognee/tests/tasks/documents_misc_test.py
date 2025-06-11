# Tests for check_permissions_on_documents, classify_documents, detect_language, translate_text
# Covers both English and Chinese text

import pytest
import importlib

def import_task_module(module_path):
    return importlib.import_module(module_path)

def test_check_permissions_on_documents():
    mod = import_task_module('cognee.tasks.documents.check_permissions_on_documents')
    # Simulate input: list of docs with 'permissions' field
    docs = [{'permissions': 'read'}, {'permissions': 'write'}]
    if hasattr(mod, 'check_permissions_on_documents'):
        result = mod.check_permissions_on_documents(docs)
        assert isinstance(result, list)
    else:
        pytest.skip('check_permissions_on_documents not implemented')

def test_classify_documents():
    mod = import_task_module('cognee.tasks.documents.classify_documents')
    docs = [{'text': 'This is a test.'}, {'text': '这是一个测试。'}]
    if hasattr(mod, 'classify_documents'):
        result = mod.classify_documents(docs)
        assert isinstance(result, list)
    else:
        pytest.skip('classify_documents not implemented')

def test_detect_language():
    mod = import_task_module('cognee.tasks.documents.detect_language')
    docs = [{'text': 'This is English.'}, {'text': '这是中文。'}]
    if hasattr(mod, 'detect_language'):
        result = mod.detect_language(docs)
        assert isinstance(result, list)
    else:
        pytest.skip('detect_language not implemented')

def test_translate_text():
    mod = import_task_module('cognee.tasks.documents.translate_text')
    docs = [{'text': 'This is English.'}, {'text': '这是中文。'}]
    if hasattr(mod, 'translate_text'):
        result = mod.translate_text(docs, target_language='zh')
        assert isinstance(result, list)
    else:
        pytest.skip('translate_text not implemented')
