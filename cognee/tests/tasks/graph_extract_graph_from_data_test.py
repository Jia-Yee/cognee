# Tests for extract_graph_from_data in tasks/graph
# Covers both English and Chinese text

import pytest
import importlib
import asyncio
import os
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk

@pytest.fixture(autouse=True)
def patch_embedding(monkeypatch):
    # Patch the embedding function to always return a vector of 4096 floats
    def dummy_embed(*args, **kwargs):
        return [float(i) for i in range(4096)]
    monkeypatch.setattr(
        "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter.embed_text",
        dummy_embed
    )

# Set LLM environment variables for testing if not already set
if "LLM_API_KEY" not in os.environ:
    os.environ["LLM_API_KEY"] = "ollama"
    os.environ["LLM_MODEL"] = "qwen2.5:latest"
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_ENDPOINT"] = "http://192.168.1.4:11434/v1"

@pytest.mark.asyncio
async def test_extract_graph_from_data_english_and_chinese():
    extract_graph = importlib.import_module('cognee.tasks.graph.extract_graph_from_data')
    english_chunk = DocumentChunk(
        document_id='eng1',
        chunk_id='eng1',
        text="This is a test about Paris and the Eiffel Tower.",
        chunk_size=1,
        chunk_index=0,
        cut_type="sentence",
        is_part_of={
            "name": "",
            "raw_data_location": "",
            "external_metadata": "",
            "mime_type": ""
        }
    )
    chinese_chunk = DocumentChunk(
        document_id='zh1',
        chunk_id='zh1',
        text="这是一个关于北京和长城的测试。",
        chunk_size=1,
        chunk_index=0,
        cut_type="sentence",
        is_part_of={
            "name": "",
            "raw_data_location": "",
            "external_metadata": "",
            "mime_type": ""
        }
    )
    try:
        from cognee.shared.data_models import KnowledgeGraph
        graph_model = KnowledgeGraph
    except ImportError:
        from pydantic import BaseModel
        graph_model = BaseModel
    result = await extract_graph.extract_graph_from_data([english_chunk], graph_model)
    assert result is not None
    assert isinstance(result, list)
    result2 = await extract_graph.extract_graph_from_data([chinese_chunk], graph_model)
    assert result2 is not None
    assert isinstance(result2, list)

# More tests for other graph functions can be added here.
