import os
from typing import List, Optional

# 1) Install needed libraries:
#    pip install groq requests langchain pydantic

from groq import Groq
from pydantic import BaseModel, Field
from langchain.llms.base import LLM

class OfficialGroqLLM(LLM, BaseModel):
    """
    A LangChain-compatible LLM wrapper that calls Groq's official Python client.
    """

    api_key: str = Field(..., description="Your Groq API key")
    model: str = Field("llama-3.3-70b-versatile", description="Groq model name")
    temperature: float = Field(0.4, description="Sampling temperature")
    max_tokens: int = Field(500, description="Maximum tokens to generate")

    @property
    def _llm_type(self) -> str:
        return "groq_official"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Sends a single 'user' message to Groq. If you need a 'system' message,
        incorporate it into the prompt or customize messages below.
        """
        # Initialize the Groq client
        client = Groq(api_key=self.api_key)

        # Convert the LangChain prompt into Groq's chat format
        # You can add a system message if desired:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        # Call the official Groq API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
            # Groq might support additional parameters if needed
        )

        # Return the text from the first choice
        return chat_completion.choices[0].message.content

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError("Async method not implemented for OfficialGroqLLM.")