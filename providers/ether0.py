"""Ether0 provider client for FutureHouse's chemistry model."""

from typing import Any, Dict

from .base import SyncProviderClient, ModelConfig


SUPPORTED_MODELS = {"ether0": ModelConfig(None, "#4CAF50")}  # Green for chemistry


class Ether0ProviderClient(SyncProviderClient):
    """Provider client for Ether0 using OpenAI-compatible API.
    
    Ether0 is FutureHouse's 24B chemistry model deployed on Modal
    with an OpenAI SDK-compatible endpoint.
    
    Usage:
        client = Ether0ProviderClient()
        result = client.query("What is the SMILES for water?")
    """
    
    # Base URL for OpenAI SDK (endpoint exposes /v1/chat/completions)
    BASE_URL = "https://themurtazanazir--ether0-inference-web.modal.run/v1"
    SUPPORTED_MODELS = SUPPORTED_MODELS
    
    def __init__(self, base_url: str | None = None):
        """Initialize client.
        
        Args:
            base_url: Override the default Modal endpoint URL (should end with /v1)
        """
        self.base_url = base_url or self.BASE_URL
        
        # Initialize OpenAI client
        from openai import OpenAI
        self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
    
    def get_model_name(self) -> str:
        return "ether0"
    
    def query(self, prompt: str, max_tokens: int = 10000, temperature: float = 0.0) -> Dict[str, Any]:
        """Query Ether0 model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 for deterministic)
            
        Returns:
            Dict with raw_answer, prompt_tokens, completion_tokens
        """
        response = self.client.chat.completions.create(
            model="ether0",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return {
            "raw_answer": response.choices[0].message.content or "",
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
