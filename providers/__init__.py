"""Provider clients for LLM APIs.

Each provider has its own module with:
- ThinkingLevel enum (if applicable)
- SUPPORTED_MODELS dict
- ProviderClient subclass (batch) or SyncProviderClient subclass (real-time)
"""

from .base import (
    BaseThinkingLevel,
    ProviderClient,
    SyncProviderClient,
    BatchStatus,
    RequestCounts,
)

from .openai import (
    OpenAIProviderClient,
    ThinkingLevel as OpenAIThinkingLevel,
    SUPPORTED_MODELS as OPENAI_MODELS,
)

from .google import (
    GoogleProviderClient,
    ThinkingLevel as GoogleThinkingLevel,
    SUPPORTED_MODELS as GOOGLE_MODELS,
)

from .anthropic import (
    AnthropicProviderClient,
    SUPPORTED_MODELS as ANTHROPIC_MODELS,
)

from .ether0 import (
    Ether0ProviderClient,
    SUPPORTED_MODELS as ETHER0_MODELS,
)


# Batch API providers
PROVIDER_MAP = {
    'openai': OpenAIProviderClient,
    'google': GoogleProviderClient,
    'anthropic': AnthropicProviderClient,
}

# Synchronous (real-time) providers
SYNC_PROVIDER_MAP = {
    'ether0': Ether0ProviderClient,
}


__all__ = [
    # Base
    "BaseThinkingLevel",
    "ProviderClient",
    "SyncProviderClient",
    "BatchStatus",
    "RequestCounts",
    # OpenAI
    "OpenAIProviderClient",
    "OpenAIThinkingLevel",
    "OPENAI_MODELS",
    # Google
    "GoogleProviderClient",
    "GoogleThinkingLevel", 
    "GOOGLE_MODELS",
    # Anthropic
    "AnthropicProviderClient",
    "ANTHROPIC_MODELS",
    # Ether0
    "Ether0ProviderClient",
    "ETHER0_MODELS",
    # Config
    "PROVIDER_MAP",
    "SYNC_PROVIDER_MAP",
]
