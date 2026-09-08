import json
import logging
from typing import List, Dict, Any, Optional

import requests

import config
from src.utils.retry import retry_with_backoff
from src.utils.cache import default_cache

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Google AI Studio Gemini client through Google's OpenAI-compatible endpoint.

    This is intentionally separate from OpenRouterClient so OpenRouter remains
    available as a future fallback without being used by the active agent path.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.model = model or config.GEMINI_MODEL
        self.base_url = (base_url or config.GEMINI_BASE_URL).rstrip("/")
        self.endpoint = f"{self.base_url}/chat/completions"

    def generate(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = False,
        temperature: float = 0.1,
        use_cache: bool = True,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not configured in .env")

        cache_key = json.dumps(
            {
                "provider": "gemini",
                "messages": messages,
                "model": self.model,
                "json_mode": json_mode,
            },
            sort_keys=True,
        )
        if use_cache:
            cached_response = default_cache.get("llm", cache_key)
            if cached_response is not None:
                return cached_response

        result = self._call(messages, json_mode, temperature)
        if use_cache:
            default_cache.set("llm", cache_key, result)
        return result

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def _call(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool,
        temperature: float,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=45)

        # Compatibility fallback for models/configurations that reject the
        # OpenAI response_format field.
        if json_mode and response.status_code == 400 and "response_format" in response.text:
            payload.pop("response_format", None)
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=45)

        if response.status_code == 429:
            raise RuntimeError(f"Gemini rate limited (429): {response.text}")
        if response.status_code in (401, 403):
            raise RuntimeError(f"Gemini auth/quota error ({response.status_code}): {response.text}")
        if response.status_code != 200:
            raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]
