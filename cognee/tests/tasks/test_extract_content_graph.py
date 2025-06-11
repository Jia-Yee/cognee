import os
import pytest
from pydantic import BaseModel
from unittest.mock import patch, MagicMock, AsyncMock

# Set LLM environment variables for testing if not already set
if "LLM_API_KEY" not in os.environ:
    os.environ["LLM_API_KEY"] = "ollama"
    os.environ["LLM_MODEL"] = "qwen2.5:latest"
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_ENDPOINT"] = "http://192.168.1.4:11434/v1"

from cognee.modules.data.extraction.knowledge_graph.extract_content_graph import extract_content_graph

class DummyGraphModel(BaseModel):
    nodes: list = []
    edges: list = []

@pytest.mark.asyncio
async def test_extract_content_graph_english_and_chinese():
    dummy_graph = DummyGraphModel(nodes=["A", "B"], edges=[("A", "B")])

    with patch("cognee.modules.data.extraction.knowledge_graph.extract_content_graph.get_llm_client") as mock_llm_client, \
         patch("cognee.modules.data.extraction.knowledge_graph.extract_content_graph.get_llm_config") as mock_llm_config, \
         patch("cognee.modules.data.extraction.knowledge_graph.extract_content_graph.render_prompt") as mock_render_prompt:

        mock_llm = MagicMock()
        mock_llm.acreate_structured_output = AsyncMock(return_value=dummy_graph)
        mock_llm_client.return_value = mock_llm

        mock_llm_config.return_value = type("Config", (), {"graph_prompt_path": "/tmp/prompt.txt"})()
        mock_render_prompt.return_value = "system prompt"

        # English
        result_en = await extract_content_graph("Paris and the Eiffel Tower.", DummyGraphModel)
        assert isinstance(result_en, DummyGraphModel)
        assert result_en.nodes == ["A", "B"]
        assert result_en.edges == [("A", "B")]

        # Chinese
        result_zh = await extract_content_graph("北京和长城。", DummyGraphModel)
        assert isinstance(result_zh, DummyGraphModel)
        assert result_zh.nodes == ["A", "B"]
        assert result_zh.edges == [("A", "B")]
