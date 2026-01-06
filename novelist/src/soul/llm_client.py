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

    async def generate_content(self, prompt: str, model_override: Optional[str] = None) -> Any:
        # NOTE: Returns GenerationResponse, typed as Any to avoid circular imports if schemas not available
        from src.contracts.schemas import GenerationResponse, TokenUsage

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
                content = ""
                usage = TokenUsage()

                if provider_stats.name == "gemini":
                    coro = asyncio.to_thread(
                        lambda: client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                        )
                    )
                    # Add timeout to prevent hanging
                    response = await asyncio.wait_for(coro, timeout=60.0)
                    content = response.text
                    
                    # Estimate tokens for Gemini if not provided (Gemini often provides it differently)
                    # For now, simple char approximation: 4 chars ~= 1 token
                    prompt_tok = len(prompt) // 4
                    comp_tok = len(content) // 4
                    usage = TokenUsage(
                        prompt_tokens=prompt_tok,
                        completion_tokens=comp_tok,
                        total_tokens=prompt_tok + comp_tok,
                        cost_usd=self._calculate_cost(provider_stats.name, model_name, prompt_tok, comp_tok)
                    )

                elif provider_stats.name in ["groq", "cerebras"]:
                    response = await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_name,
                    )
                    content = response.choices[0].message.content
                    
                    if response.usage:
                        usage = TokenUsage(
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                            cost_usd=self._calculate_cost(
                                provider_stats.name, 
                                model_name, 
                                response.usage.prompt_tokens, 
                                response.usage.completion_tokens
                            )
                        )
                
                return GenerationResponse(
                    content=content,
                    usage=usage,
                    model_name=model_name,
                    provider=provider_stats.name
                )

            except Exception as e:
                error_str = str(e)
                # Handle rate limits
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                
                if is_rate_limit:
                    print(f"[WARN] Rate limit on {provider_stats.name}. Failing over...")
                    # Temporarily back off this provider
                    provider_stats.last_request_time = time.time() + 10.0 
                else:
                    print(f"[ERROR] Error on {provider_stats.name}: {e}")
                
                last_error = e
                continue # Try next provider

        print(f"[ERROR] All providers failed. Last error: {last_error}")
        return None

    def _resolve_model(self, provider: str, override: Optional[str]) -> str:
        """Get the correct model name for the provider."""
        # Simple override logic for now
        if override:
             # Basic mapping if user requests "flash" or "pro" abstractly
            if "flash" in override.lower():
                if provider == "gemini": return "gemini-2.0-flash"
                if provider == "groq": return "llama-3.3-70b-versatile" # Fast equivalent
                if provider == "cerebras": return "llama-3.1-8b"
            if "pro" in override.lower():
                if provider == "gemini": return "gemini-2.0-pro" # Placeholder if exists
                if provider == "groq": return "llama-3.3-70b-versatile"
        
        if provider == "gemini":
            return "gemini-2.0-flash"
        elif provider == "groq":
            return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        elif provider == "cerebras":
            return os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
        return "llama-3.3-70b"

    def _calculate_cost(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost (very rough estimates)."""
        # Pricing per 1M tokens (fake values for demo/beta)
        # Gemini Flash 2.0 (Preview is free, but let's assume pricing like 1.5 Flash)
        # Input: $0.075 / 1M, Output: $0.30 / 1M
        pricing = {
            "gemini": {"in": 0.10, "out": 0.40}, 
            "groq": {"in": 0.50, "out": 0.50}, # Llama 70b approx
            "cerebras": {"in": 0.20, "out": 0.20},
        }
        
        p = pricing.get(provider, {"in": 0.0, "out": 0.0})
        cost = (prompt_tokens / 1_000_000 * p["in"]) + (completion_tokens / 1_000_000 * p["out"])
        return round(cost, 6)