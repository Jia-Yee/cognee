import pytest
import asyncio
from cognee.modules.retrieval.function_calling_completion_retriever import FunctionCallingCompletionRetriever

@pytest.mark.asyncio
async def test_add_function_call():
    retriever = FunctionCallingCompletionRetriever()
    query = "add([10, 20, 30])"
    result = await retriever.get_completion(query)
    assert "Function call: add([10, 20, 30])" in result[0]
    assert "Answer:" in result[0]

@pytest.mark.asyncio
async def test_entropy_weight_function_call():
    retriever = FunctionCallingCompletionRetriever()
    query = "Use entropy_weight to calculate weights for [[1,2,3],[4,5,6],[7,8,9]]"
    result = await retriever.get_completion(query)
    assert "Function call: entropy_weight" in result[0]
    assert "Answer:" in result[0]

@pytest.mark.asyncio
async def test_no_function_call():
    retriever = FunctionCallingCompletionRetriever()
    query = "What is the capital of France?"
    result = await retriever.get_completion(query)
    assert "LLM completion" in result[0]
