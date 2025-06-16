import logging
from typing import Any, Optional

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.llm.get_llm_client import get_llm_client  # Make sure this import is correct
from cognee.modules.retrieval.utils.completion import generate_completion, generate_completion_with_function_calling
from cognee.infrastructure.llm.prompts import render_prompt

import numpy as np
import pandas as pd
import re
import ast
import json

logger = logging.getLogger(__name__)

tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Sum a list of numbers. Example: add([1,2,3]) returns 6.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Numbers to sum"
                    }
                },
                "required": ["data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "entropy_weight",
            "description": "Calculate entropy-based weights for a matrix of data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Matrix of numbers"
                    }
                },
                "required": ["data"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_city_score",
            "description": "Calculate the international communication influence score for cities based on indicators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator_data": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Matrix of city indicators"
                    }
                },
                "required": ["indicator_data"]
            }
        }
    }
]

level_1_tags = [
    "网络传播影响力",
    "媒体报道影响力",
    "社交媒体影响力",
    "搜索引擎影响力",
    "国际访客影响力",
]

weights_dict = {
    "网络传播":0.1523,
    # 网络传播影响力 (官方社交媒体)
    "媒体报道":0.183,
    # 媒体报道影响力 (国内外文报道-国内)
    "社交媒体":0.1796,
    # 社交媒体影响力 (Facebook关键词词频)
    "搜索引擎":0.2152,
    # 搜索引擎影响力
    "国际访客": 0.2699,
    # 国际访客影响力
}


level_1_keys = [
    "网络传播",
    "媒体报道",
    "社交媒体",
    "搜索引擎",
    "国际访客",
]

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
    return weights.values.tolist()

@register_function("calculate_city_score")
def calculate_city_score(indicator_data):
    """
    计算城市国际传播影响力得分
    :param indicator_data: DataFrame, 行是城市，列是底层指标
    :return: 最终得分数组 (0-100范围)
    """
    if not isinstance(indicator_data, pd.DataFrame):
        indicator_data = pd.DataFrame(indicator_data)
        indicator_data.columns = level_1_keys  # Ensure columns match level_1_tags
    # 1. 对每个底层指标进行min-max标准化
    normalized_indicators = (indicator_data - indicator_data.min()) / (
        indicator_data.max() - indicator_data.min() + 1e-8)
    
    # 2. 计算加权原始总分
    weighted_scores = []
    for col in normalized_indicators.columns:
        if col in weights_dict:
            weighted_scores.append(normalized_indicators[col] * weights_dict[col])
    
    total_scores = pd.concat(weighted_scores, axis=1).sum(axis=1)
    
    # 3. 对原始总分进行min-max标准化得到最终得分
    final_scores = (total_scores - total_scores.min()) / (
        total_scores.max() - total_scores.min() )
    final_scores_percent = np.round(final_scores * 100, 2)
    return final_scores_percent.values


@register_function("add")
def add(data):
    """Sum a list of numbers or add multiple arguments."""
    if isinstance(data, (list, tuple)):
        return sum(data)
    return data

async def call_llm_to_decide_function(query: str, context: str, available_functions: list) -> dict:
    """
    Call the LLM (via generate_completion) to decide which function to use and extract parameters.
    Returns a dict: {"function": function_name, "parameters": ...}
    """
    logger.info(f"[FunctionCalling] Deciding function for query: {query} with context: {context}")
    llm_response = await generate_completion_with_function_calling(
        query=query,
        context=context,
        user_prompt_path="function_call_decision.txt",
        system_prompt_path="function_call_decision_system.txt",
        tools=tools,
    )

    logger.info(f"[FunctionCalling] LLM response for function decision: {llm_response}")
    
    try:
        cleaned_response = llm_response['content'].strip("```json\n").strip("```").strip()
        decision = json.loads(cleaned_response)
        logger.info(f"[FunctionCalling] Parsed function decision: {decision}")
        return decision
    except Exception as e:
        logger.error(f"[FunctionCalling] Failed to parse LLM response: {llm_response}, error: {e}")
        return {"function": None, "parameters": None}

async def call_llm_generate_answer(query: str, context: str) -> str:
    """
    Call the LLM (via generate_completion) to generate a natural language answer.
    """
    args = {"question": query, "context": context}
    user_prompt = render_prompt("function_call_answer.txt", args)
    answer = await generate_completion(
        query=user_prompt,
        context="",
        user_prompt_path="function_call_answer.txt",
        system_prompt_path="function_call_answer_system.txt",
    )
    logger.info(f"[FunctionCalling] LLM answer: {answer}")
    if isinstance(answer, dict) and "content" in answer:
        answer = answer["content"]
    return answer

async def generate_function_calling_completion(query: str, context: str, user_prompt_path: str, system_prompt_path: str) -> Any:
    """
    LLM-driven function calling: LLM decides which function to use and extracts parameters.
    Runs the function, updates the context with the result, and generates a final answer.
    """
    logger.info(f"[FunctionCalling] Start function-calling completion for query: {query}")
    available_functions = list(function_registry.keys())
    llm_decision = await call_llm_to_decide_function(query, context, available_functions)
    func_name = llm_decision.get("function")
    params = llm_decision.get("parameters")
    logger.info(f"[FunctionCalling] LLM decided function: {func_name}, params: {params}")
    if func_name and func_name in function_registry:
        func = function_registry[func_name]
        try:
            
            if func_name == "entropy_weight":
                result = func(params["data"])
                logger.info(f"[FunctionCalling] Function result: {result}")
                if len(level_1_tags) != len(result):
                    result_str = f"一级指标权重： {func_name} = {result}"
                else:
                    result_str = f"一级指标权重：{dict(zip(level_1_tags, result))}"
            elif func_name == "calculate_city_score":
                result = func(params["indicator_data"])
                logger.info(f"[FunctionCalling] Function result: {result}")
                if len(llm_decision['city_list']) != len(result):
                    result_str = f"城市国际传播影响力得分: {llm_decision['city_list']} {result}"
                else:
                    result_str = f"城市国际传播影响力得分：{dict(zip(llm_decision['city_list'], result))}"
            else:
                result_str = f"result: {result}"
            logger.info(f"[FunctionCalling] Result string: {result_str}")
            updated_context = result_str + "\n" + context
            answer = await call_llm_generate_answer(query, updated_context)
            logger.info(f"[FunctionCalling] Final answer: {answer}")
            return f"{answer}"
        except Exception as e:
            logger.error(f"[FunctionCalling] Function call error: {func_name}({params}) - {e}")
            return f"Function call error: {func_name}({params}) - {e}"
    logger.info(f"[FunctionCalling] No function called, fallback to normal completion.")
    return f"LLM completion: {query} | Context: {context}"

class FunctionCallingCompletionRetriever(BaseRetriever):
    """
    Retriever for LLM-based completion with function calling (add tool, entropy_weight, etc).
    Public methods:
    - get_context(query: str) -> str
    - get_completion(query: str, context: Optional[Any] = None) -> Any
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
        """
        Retrieves relevant document chunks as context (same as CompletionRetriever).
        """
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
        """
        Generates a function-calling LLM completion using the context.
        """
        if context is None:
            context = await self.get_context(query)
        completion = await generate_function_calling_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
        )
        return [completion]
