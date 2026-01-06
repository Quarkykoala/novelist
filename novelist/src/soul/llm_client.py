"""Unified LLM client abstraction for Gemini, Groq, and Cerebras with Rate Limiting."""

import os
import asyncio
import re
import time
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Unified client for multiple LLM providers with built-in Rate Limiting."""

    # Class-level lock to ensure rate limits across all instances
    _rate_limit_lock = asyncio.Lock()
    _last_request_time = 0.0

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model = model
        
        self._gemini_client = None
        self._groq_client = None
        self._cerebras_client = None

    async def _get_gemini_client(self):
        if self._gemini_client is None:
            import google.genai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    async def _get_groq_client(self):
        if self._groq_client is None:
            from groq import AsyncGroq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not set")
            self._groq_client = AsyncGroq(api_key=api_key)
        return self._groq_client

    async def _get_cerebras_client(self):
        if self._cerebras_client is None:
            from cerebras.cloud.sdk import AsyncCerebras
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError("CEREBRAS_API_KEY not set")
            self._cerebras_client = AsyncCerebras(api_key=api_key)
        return self._cerebras_client

    async def _throttle(self):
        """Enforce rate limits based on provider."""
        # Gemini free tier is ~5 RPM (12s interval)
        # Groq/Cerebras free tier is ~30 RPM (2s interval)
        if self.provider == "gemini":
            interval = 12.0
        else:
            interval = 2.0
        
        async with self._rate_limit_lock:
            now = time.time()
            elapsed = now - LLMClient._last_request_time
            if elapsed < interval:
                wait_time = interval - elapsed
                await asyncio.sleep(wait_time)
            LLMClient._last_request_time = time.time()

    async def generate_content(self, prompt: str, model_override: Optional[str] = None, retries: int = 5) -> Optional[str]:
        """Generate content with built-in retries and throttling."""
        target_model = model_override or self.model
        
        for attempt in range(retries):
            try:
                await self._throttle()
                
                if self.provider == "gemini":
                    client = await self._get_gemini_client()
                    if not target_model or "llama" in target_model.lower() or "mixtral" in target_model.lower():
                        model_name = "gemini-2.0-flash"
                    else:
                        model_name = target_model
                    
                    response = await asyncio.to_thread(
                        lambda: client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                        )
                    )
                    return response.text

                elif self.provider == "groq":
                    client = await self._get_groq_client()
                    if not target_model or "gemini" in target_model.lower():
                        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
                    else:
                        model_name = target_model
                    
                    response = await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                    )
                    return response.choices[0].message.content

                elif self.provider == "cerebras":
                    client = await self._get_cerebras_client()
                    if not target_model or "gemini" in target_model.lower():
                        model_name = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
                    else:
                        model_name = target_model
                    
                    response = await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                    )
                    return response.choices[0].message.content

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = 10.0 * (2 ** attempt)
                    match = re.search(r"retry in (\d+(\.\d+)?)s", error_str)
                    if match:
                        delay = float(match.group(1)) + 1.0
                    print(f"[WARN] Rate limit hit. Waiting {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"[ERROR] Generation failed: {e}")
                    return None
        
        return None
