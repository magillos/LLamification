#!/usr/bin/env python3
"""Test model alias resolution."""

import asyncio
from llamification.proxy.server import ProxyServer
from llamification.proxy.providers import get_provider

async def test_alias():
    print("=" * 60)
    print("Testing Model Alias Resolution")
    print("=" * 60)
    
    # Simulate configuration with a generic OpenAI-compatible custom provider.
    provider = get_provider("custom:cp1", "dummy_key", "https://example.com/v1")
    server = ProxyServer()
    
    # Configure with models (as if fetched from API)
    models = [
        "z-ai/glm-5.1",
        "meta/llama-3.1-8b-instruct",
        "google/gemma-2-2b-it",
    ]
    active_model = "z-ai/glm-5.1"
    
    server.configure(provider, models, active_model)
    
    print(f"\n✓ Configured with {len(models)} models:")
    for m in models:
        print(f"   - {m}")
    
    print(f"\n✓ Active model: {active_model}")
    
    print(f"\n✓ Generated aliases:")
    for alias, full_id in server._model_aliases.items():
        print(f"   '{alias}' -> '{full_id}'")
    
    print("\n" + "=" * 60)
    print("Testing Alias Resolution")
    print("=" * 60)
    
    test_cases = [
        "glm-5.1",  # Short name (alias)
        "z-ai/glm-5.1",  # Full name
        "llama-3.1-8b-instruct",  # Another alias
        "default",  # Special case
        "unknown-model",  # Not in list
    ]
    
    for test_model in test_cases:
        resolved = server._resolve_model_alias(test_model)
        if test_model == resolved:
            print(f"   '{test_model}' -> (no change)")
        else:
            print(f"   '{test_model}' -> '{resolved}' ✓")
    
    print("\n" + "=" * 60)
    print("How This Helps You")
    print("=" * 60)
    print("\n✅ You can now keep your environment variable as:")
    print("   export AURSCAN_OPENAI_MODEL=\"glm-5.1\"")
    print("\n✅ LLamification will automatically resolve it to:")
    print("   z-ai/glm-5.1")
    print("\n✅ Works just like LiteLLM's model aliasing!")

if __name__ == "__main__":
    asyncio.run(test_alias())
