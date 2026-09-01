"""
Gemini LLM client wrapper for RLM.
"""

from __future__ import annotations
import os
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    """Thin wrapper around the Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. "
                "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )

        genai.configure(api_key=self.api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Call the Gemini generate content endpoint.

        Args:
            messages: Either a list of message dicts (OpenAI format) or a plain string.
            max_tokens: Optional token limit for the completion.

        Returns:
            The model's response text.
        """
        try:
            # Convert OpenAI message format to Gemini format
            if isinstance(messages, str):
                prompt = messages
            elif isinstance(messages, list):
                # Simple conversion for RLM's usage
                # Root RLM expects system prompt + user prompts
                # Sub RLM expects single prompt
                # We'll join them or use them as a conversation history
                history = []
                last_msg = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        # Gemini often prefers system instructions in the model config,
                        # but for simplicity we'll prepend it if it's the first message.
                        last_msg += f"System: {content}\n\n"
                    elif role == "user":
                        last_msg += content
                    elif role == "assistant":
                        history.append({"role": "user", "parts": [last_msg]})
                        history.append({"role": "model", "parts": [content]})
                        last_msg = ""
                
                prompt = last_msg if last_msg else "Continue"
                
                if history:
                    chat = self.model.start_chat(history=history)
                    response = chat.send_message(prompt)
                else:
                    response = self.model.generate_content(prompt)
            else:
                response = self.model.generate_content(str(messages))

            return response.text

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
