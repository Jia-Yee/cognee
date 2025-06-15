from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError

import numpy as np
import pandas as pd
import re
import ast
import json

# Function registry for user-defined tools
function_registry = {}

def register_function(name):
    def decorator(func):
        function_registry[name] = func
        return func
    return decorator

@register_function("entropy_weight")
def entropy_weight(data):
    """
    熵值法确定一级指标权重
    :param data: DataFrame-like (list of lists, or pandas DataFrame), 行是城市，列是统计计数
    :return: 权重数组, 熵值数组
    """
    # Accept list of lists or DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    # 1. 数据标准化 (min-max)
    normalized_data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    # 2. 计算比重
    p = normalized_data / (normalized_data.sum() + 1e-8)
    # 3. 计算熵值
    n = len(data)
    k = 1 / np.log(n + 1e-8)
    entropy = -k * (p * np.log(p + 1e-8)).sum()
    # 4. 计算权重
    d = 1 - entropy
    weights = np.round(d / d.sum(), 4)
    return weights.values.tolist(), entropy.values.tolist()

@register_function("add")
def add(data):
    """Sum a list of numbers or add multiple arguments."""
    if isinstance(data, (list, tuple)):
        return sum(data)
    return data

async def call_llm_to_decide_function(query: str, context: str, available_functions: list) -> dict:
    """
    Call the LLM to decide which function to use and extract parameters.
    Returns a dict: {"function": function_name, "parameters": ...}
    """
    # Prepare function descriptions
    function_descriptions = {
        "add": "add(data): Sum a list of numbers. Example: add([1,2,3]) returns 6.",
        "entropy_weight": (
            "entropy_weight(data): Calculate entropy-based weights for a matrix of data. "
            "Input is a list of lists (rows: samples, columns: features). "
            "Example: entropy_weight([[1,2,3],[4,5,6],[7,8,9]])"
        ),
    }
    # Compose the prompt for the LLM
    prompt = f"""
You are an intelligent assistant that can decide whether to call a function (tool) to help answer a user's query.
Here are the available functions/tools:
{json.dumps(function_descriptions, indent=2)}

Given the following user query and context, decide if a function call is needed.
If so, return a JSON object with the function name and parameters to use.
If not, return: {{"function": null, "parameters": null}}

User query: {query}

Context:
{context}

Respond ONLY with a JSON object, e.g.:
{{"function": "add", "parameters": [1,2,3]}}
or
{{"function": "entropy_weight", "parameters": [[1,2,3],[4,5,6],[7,8,9]]}}
or
{{"function": null, "parameters": null}}
"""

    # --- Replace with your actual LLM call ---
    # Example using an OpenAI-like async client:
    # from your_llm_client import get_llm_client
    # llm_client = get_llm_client()
    # llm_response = await llm_client.acomplete(prompt)
    # try:
    #     decision = json.loads(llm_response)
    #     return decision
    # except Exception:
    #     return {"function": None, "parameters": None}

    # For now, fallback to mock logic if LLM is not available:
    import re
    import ast
    if "add" in query:
        match = re.search(r"add\s*\(([^)]*)\)", query)
        if match:
            arg_str = match.group(1)
            try:
                params = ast.literal_eval(arg_str)
            except Exception:
                params = [ast.literal_eval(a.strip()) for a in arg_str.split(',') if a.strip()]
                if len(params) == 1:
                    params = params[0]
            return {"function": "add", "parameters": params}
    if "entropy_weight" in query:
        match = re.search(r"entropy_weight\s*\(([^)]*)\)", query)
        if match:
            arg_str = match.group(1)
            try:
                params = ast.literal_eval(arg_str)
            except Exception:
                params = [[int(x) for x in re.findall(r"\d+", row)] for row in arg_str.split('],[')]
            return {"function": "entropy_weight", "parameters": params}
        match = re.search(r"\[\[.*?\]\]", query)
        if match:
            try:
                params = ast.literal_eval(match.group(0))
                return {"function": "entropy_weight", "parameters": params}
            except Exception:
                pass

    return {"function": None, "parameters": None}

async def call_llm_generate_answer(query: str, context: str) -> str:
    """
    Call the LLM to generate a natural language answer given the query and updated context.
    Replace this with your actual LLM call.
    """
    # Example prompt for the LLM:
    prompt = f"""
    Given the following context and user query, answer the question concisely and clearly.
    Context:
    {context}
    User query: {query}
    """
    # --- Replace with your LLM call ---
    # For demonstration, just echo a mock answer:
    return f"[LLM Answer] Based on the context, the answer to '{query}' is: {context.splitlines()[0]}"

async def generate_function_calling_completion(query: str, context: str, user_prompt_path: str, system_prompt_path: str) -> Any:
    """
    LLM-driven function calling: LLM decides which function to use and extracts parameters.
    Runs the function, updates the context with the result, and generates a final answer.
    """
    available_functions = list(function_registry.keys())
    llm_decision = await call_llm_to_decide_function(query, context, available_functions)
    func_name = llm_decision.get("function")
    params = llm_decision.get("parameters")
    if func_name and func_name in function_registry:
        func = function_registry[func_name]
        try:
            result = func(params)
            # Add the function result to the beginning of the context
            result_str = f"Function call: {func_name}({params}) = {result}"
            updated_context = result_str + "\n" + context
            # Now call the LLM to generate the final answer
            answer = await call_llm_generate_answer(query, updated_context)
            return f"{result_str}\nAnswer: {answer}"
        except Exception as e:
            return f"Function call error: {func_name}({params}) - {e}"
    # Otherwise, fallback to normal completion
    return f"LLM completion: {query} | Context: {context}"

class FunctionCallingCompletionRetriever(BaseRetriever):
    """
    Retriever for LLM-based completion with function calling (add tool).
    """
    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        top_k: Optional[int] = 1,
    ):
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.top_k = top_k if top_k is not None else 1

    async def get_context(self, query: str) -> str:
        vector_engine = get_vector_engine()
        try:
            found_chunks = await vector_engine.search("DocumentChunk_text", query, limit=self.top_k)
            if len(found_chunks) == 0:
                return ""
            chunks_payload = [found_chunk.payload["text"] for found_chunk in found_chunks]
            return "\n".join(chunks_payload)
        except CollectionNotFoundError as error:
            raise NoDataError("No data found in the system, please add data first.") from error

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        if context is None:
            context = await self.get_context(query)
        completion = await generate_function_calling_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )
        return [completion]
