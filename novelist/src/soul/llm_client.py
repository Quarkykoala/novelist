"""Smart LLM client with provider failover and adaptive throttling."""

import os
import asyncio
import re
import time
import random
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

class ProviderStats:
    """Track usage stats per provider."""
    def __init__(self, name: str, interval: float):
        self.name = name
        self.interval = interval
        self.last_request_time = 0.0
        self.lock = asyncio.Lock()

class LLMClient:
    """Smart client that fails over between providers to maximize throughput."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        # Default provider preference order
        self.providers = [
            ProviderStats("cerebras", 2.0), # 30 RPM
            ProviderStats("groq", 2.0),     # 30 RPM
            ProviderStats("gemini", 12.0),  # 5 RPM
        ]
        
        # Move requested provider to front if specified
        requested = provider or os.getenv("LLM_PROVIDER", "cerebras").lower()
        self.providers.sort(key=lambda p: p.name == requested, reverse=True)
        
        self.model = model
        
        self._clients = {}

    async def _get_client(self, provider: str):
        if provider not in self._clients:
            if provider == "gemini":
                import google.genai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    self._clients[provider] = genai.Client(api_key=api_key)
            elif provider == "groq":
                from groq import AsyncGroq
                api_key = os.getenv("GROQ_API_KEY")
                if api_key:
                    self._clients[provider] = AsyncGroq(api_key=api_key)
            elif provider == "cerebras":
                from cerebras.cloud.sdk import AsyncCerebras
                api_key = os.getenv("CEREBRAS_API_KEY")
                if api_key:
                    self._clients[provider] = AsyncCerebras(api_key=api_key)
        
        return self._clients.get(provider)

    async def _throttle(self, provider_stats: ProviderStats):
        """Enforce rate limits for a specific provider."""
        async with provider_stats.lock:
            now = time.time()
            elapsed = now - provider_stats.last_request_time
            if elapsed < provider_stats.interval:
                wait_time = provider_stats.interval - elapsed
                # Add jitter to prevent thundering herd
                wait_time += random.uniform(0.1, 0.5)
                await asyncio.sleep(wait_time)
            provider_stats.last_request_time = time.time()

    async def generate_content(self, prompt: str, model_override: Optional[str] = None) -> Optional[str]:
        """Try to generate content using available providers in order."""
        
        last_error = None

        for provider_stats in self.providers:
            client = await self._get_client(provider_stats.name)
            if not client:
                continue

            try:
                # Resolve model name for this provider
                model_name = self._resolve_model(provider_stats.name, model_override)
                
                # Throttle before request
                await self._throttle(provider_stats)
                
                # Execute request
                if provider_stats.name == "gemini":
                    response = await asyncio.to_thread(
                        lambda: client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                        )
                    )
                    return response.text

                elif provider_stats.name in ["groq", "cerebras"]:
                    response = await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                    )
                    return response.choices[0].message.content

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                
                if is_rate_limit:
                    print(f"[WARN] Rate limit on {provider_stats.name}. Failing over...")
                    # Temporarily back off this provider by updating its last request time to future
                    provider_stats.last_request_time = time.time() + 10.0 
                else:
                    print(f"[ERROR] Error on {provider_stats.name}: {e}")
                
                last_error = e
                continue # Try next provider

        print(f"[ERROR] All providers failed. Last error: {last_error}")
        return None

    def _resolve_model(self, provider: str, override: Optional[str]) -> str:
        """Get the correct model name for the provider."""
        if override:
            # If override looks like a specific provider model, try to map it
            # But mostly we trust the override if it's generic
            pass

        if provider == "gemini":
            return "gemini-2.0-flash"
        elif provider == "groq":
            return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        elif provider == "cerebras":
            return os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
        return "llama-3.3-70b"