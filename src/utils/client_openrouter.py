import json
import requests
import itertools
import importlib
from typing import List, Dict, Any, Optional
import config
from src.utils.retry import retry_with_backoff
from src.utils.cache import default_cache

class OpenRouterClient:
    """
    OpenRouter LLM Client supporting free-tier API key rotation,
    retry with exponential backoff, caching, and resilient JSON parsing.
    """
    def __init__(self, api_keys: Optional[List[str]] = None, model: Optional[str] = None):
        importlib.reload(config)
        self.api_keys = api_keys or config.OPENROUTER_API_KEYS
        self.model = model or config.OPENROUTER_MODEL
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self._key_cycle = itertools.cycle(self.api_keys) if self.api_keys else None

    def _get_next_key(self) -> Optional[str]:
        if not self._key_cycle:
            return None
        return next(self._key_cycle)

    def generate(self, messages: List[Dict[str, str]], json_mode: bool = False, temperature: float = 0.1, use_cache: bool = True) -> str:
        """
        Generate completion using rotating API keys and retries.
        """
        # Ensure keys are loaded dynamically
        if not self.api_keys:
            importlib.reload(config)
            self.api_keys = config.OPENROUTER_API_KEYS
            self._key_cycle = itertools.cycle(self.api_keys) if self.api_keys else None

        cache_key = json.dumps({"messages": messages, "model": self.model, "json_mode": json_mode}, sort_keys=True)
        if use_cache:
            cached_response = default_cache.get("llm", cache_key)
            if cached_response is not None:
                return cached_response

        if not self.api_keys:
            print("[OpenRouter Warning] No API keys configured in OPENROUTER_API_KEYS. Returning mock response.")
            return json.dumps({
                "has_enough_info": True,
                "confidence_score": 10,
                "extracted_facts": [],
                "identified_conflicts": [],
                "rationale": "Mock response because no API keys were provided.",
                "missing_information": [],
                "next_suggested_queries": []
            }) if json_mode else "Mock LLM response (no API keys configured)."

        num_keys = len(self.api_keys)
        last_exception = None

        for _ in range(num_keys):
            current_key = self._get_next_key()
            try:
                result = self._call_with_key(current_key, messages, json_mode, temperature)
                if use_cache:
                    default_cache.set("llm", cache_key, result)
                return result
            except Exception as e:
                print(f"[OpenRouter Key Error] Key starting with '{current_key[:8]}...' failed: {e}. Trying next key...")
                last_exception = e

        raise RuntimeError(f"All OpenRouter API keys failed. Last error: {last_exception}")

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def _call_with_key(self, api_key: str, messages: List[Dict[str, str]], json_mode: bool, temperature: float) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/NeuroForge/NeuroForge-Codefest2026",
            "X-Title": "NeuroForge SLIIT Codefest 2026 Assistant",
            "Content-Type": "application/json"
        }
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=45)

        # Fallback if model rejects strict response_format parameter
        if json_mode and response.status_code == 400 and "response_format" in response.text:
            del payload["response_format"]
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=45)
        
        if response.status_code == 429:
            raise RuntimeError(f"Rate limited (429): {response.text}")
        elif response.status_code in (401, 403):
            raise RuntimeError(f"Auth / Quota error ({response.status_code}): {response.text}")
        elif response.status_code != 200:
            raise RuntimeError(f"API Error ({response.status_code}): {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]
