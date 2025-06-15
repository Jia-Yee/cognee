
import os
import httpx
from typing import Type, Optional, List
from pydantic import BaseModel
from cognee.infrastructure.llm.llm_interface import LLMInterface
from openai import OpenAI
from cognee.infrastructure.llm.rate_limiter import (
    rate_limit_async,
    rate_limit_sync,
    sleep_and_retry_async,
    sleep_and_retry_sync,
)
from cognee.modules.observability.get_observe import get_observe

observe = get_observe()

class DeepSeekAdapter(LLMInterface):
    name = "DeepSeek"
    model: str
    api_key: str
    api_version: str
    MAX_RETRIES = 5

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        model: str,
        transcription_model: str,
        max_tokens: int,
        streaming: bool = False,
    ):
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint.rstrip('/')
        self.api_version = api_version
        self.max_tokens = max_tokens
        self.streaming = streaming
        self.transcription_model = transcription_model
        self.client = OpenAI(api_key=api_key, base_url=self.endpoint)

    async def send_messages(self, model, messages, tools=None):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools
        )
        return response

    @observe(as_type="generation")
    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, 
        text_input: str, 
        system_prompt: str, 
        response_model: Type[BaseModel], 
        tools: Optional[List[dict]] = None
    ) -> BaseModel:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_input}
        ]


        response = await self.send_messages(self.model, messages, tools)
        
        response.choices[0].message
        if "tool_calls" in response.choices[0].message:
            tool_call = response.choices[0].message[0]
            return {
                "tool_name": tool_call["function"]["name"],
                "arguments": tool_call["function"]["arguments"]
            }
        
        return {"content": response.choices[0].message.content}

    @observe
    @sleep_and_retry_sync()
    @rate_limit_sync
    def create_structured_output(
        self, 
        text_input: str, 
        system_prompt: str, 
        response_model: Type[BaseModel]
    ) -> BaseModel:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_input}
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.7
        }

        resp = requests.post(
            f"{self.endpoint}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        
        return response_model.parse_obj({"content": data["choices"][0]["message"]["content"]})
