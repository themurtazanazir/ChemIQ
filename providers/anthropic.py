"""Anthropic Claude provider client."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .base import BaseThinkingLevel, ProviderClient, BatchStatus, RequestCounts, ModelConfig


SUPPORTED_MODELS = {
    # Standard models
    "claude-3-5-sonnet": ModelConfig(None, "#E65100"),  # Orange 900
    "claude-3-5-haiku": ModelConfig(None, "#EF6C00"),   # Orange 800
    "claude-3-opus": ModelConfig(None, "#F57C00"),      # Orange 700
    "claude-3-sonnet": ModelConfig(None, "#FB8C00"),    # Orange 600
    "claude-3-haiku": ModelConfig(None, "#FFA726"),     # Orange 400
    
    # Claude 4 / 4.5 models (all support thinking)
    # Opus
    "claude-opus-4-20250514": ModelConfig("budget", "#BF360C"), # Deep Orange 900
    "claude-opus-4": ModelConfig("budget", "#BF360C"),          # Deep Orange 900
    "claude-opus-4.1-20250805": ModelConfig("budget", "#BF360C"), # Deep Orange 900
    "claude-opus-4.1": ModelConfig("budget", "#BF360C"),        # Deep Orange 900
    "claude-opus-4-5-20251101": ModelConfig("budget", "#D84315"), # Deep Orange 800
    "claude-opus-4-5": ModelConfig("budget", "#D84315"),        # Deep Orange 800
    
    # Sonnet
    "claude-sonnet-4-20250514": ModelConfig("budget", "#E64A19"), # Deep Orange 600
    "claude-sonnet-4": ModelConfig("budget", "#E64A19"),          # Deep Orange 600
    "claude-sonnet-4-5-20250929": ModelConfig("budget", "#F4511E"), # Deep Orange 500
    "claude-sonnet-4-5": ModelConfig("budget", "#F4511E"),        # Deep Orange 500
    
    # Haiku
    "claude-haiku-4-5-20251001": ModelConfig("budget", "#FF7043"), # Coral (Deep Orange 400)
    "claude-haiku-4-5": ModelConfig("budget", "#FF7043"),          # Coral
}


class AnthropicProviderClient(ProviderClient):
    """Provider client for Anthropic Claude models."""
    
    ThinkingLevel = None  # Anthropic uses integer budget_tokens only
    SUPPORTED_MODELS = SUPPORTED_MODELS
    ENV_VAR_NAME = "ANTHROPIC_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            from anthropic import Anthropic
            from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
            from anthropic.types.messages.batch_create_params import Request
            self.client = Anthropic(api_key=api_key)
            self._MessageCreateParamsNonStreaming = MessageCreateParamsNonStreaming
            self._Request = Request
        except ImportError as e:
            raise ImportError(
                f"Anthropic SDK not installed. Install it with: pip install anthropic\n"
                f"Original error: {e}"
            )
    
    def create_submission_file(
        self,
        questions: List[Dict[str, Any]],
        output_path: str,
        model_name: str,
        thinking: Union[int, None] = None,
        **kwargs
    ) -> str:
        """Create a batch submission file for Anthropic models."""
        # Validate
        self.validate_thinking(model_name, thinking)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        thinking_str = str(thinking) if thinking else "0"
        
        requests = []
        for question in questions:
            question_id = question["uuid"]
            prompt = question["prompt"]
            
            # Anthropic custom_id must be ^[a-zA-Z0-9_-]{1,64}$
            # We used to embed metadata, but that breaks the regex (no colons).
            # We now use just the UUID. Ensure it's safe.
            custom_id = question_id
            
            request_params = {
                "model": model_name,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "messages": [{"role": "user", "content": prompt}]
            }

            # Add thinking parameter if budget is specified
            if thinking and thinking > 0:
                request_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking
                }
                # Ensure max_tokens is larger than thinking budget
                if request_params["max_tokens"] <= thinking:
                    request_params["max_tokens"] = thinking + 4096
            
            # Store as dict with 'params' key matching Anthropic structure
            request = {"custom_id": custom_id, "params": request_params}
            
            # Add additional parameters
            for key, value in kwargs.items():
                if key not in ["max_tokens"]:
                    request["params"][key] = value
            
            requests.append(request)
        
        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for req in requests:
                f.write(json.dumps(req, ensure_ascii=False) + '\n')
        
        return output_path
    
    def submit_batch(
        self,
        submission_file: str,
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """Submit a batch job to Anthropic API."""
        batch_requests = []
        
        with open(submission_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                req = json.loads(line)
                custom_id = req["custom_id"]
                params = req["params"]
                
                message_params = self._MessageCreateParamsNonStreaming(
                    model=params["model"],
                    max_tokens=params.get("max_tokens", 4096),
                    messages=params["messages"],
                    **{k: v for k, v in params.items() if k not in ["model", "max_tokens", "messages"]}
                )
                
                batch_requests.append(
                    self._Request(custom_id=custom_id, params=message_params)
                )
        
        message_batch = self.client.messages.batches.create(requests=batch_requests)
        return message_batch.id
    
    def check_batch_status(self, batch_id: str) -> BatchStatus:
        """Check the status of an Anthropic batch job."""
        batch_result = self.client.messages.batches.retrieve(batch_id)
        
        status_map = {
            "pending": "pending",
            "in_progress": "in_progress",
            "completed": "completed",
            "failed": "failed"
        }
        status = status_map.get(batch_result.processing_status, batch_result.processing_status)

        request_counts = None
        if hasattr(batch_result, 'request_counts') and batch_result.request_counts:
            rc = batch_result.request_counts
            
            # Map Anthropic counts (succeeded, errored, canceled, expired, processing)
            # to our generic structure (total, completed, failed)
            
            # Helper to safely get count
            def get_cnt(obj, attr):
                if isinstance(obj, dict):
                    return obj.get(attr, 0)
                return getattr(obj, attr, 0)
            
            succeeded = get_cnt(rc, 'succeeded')
            errored = get_cnt(rc, 'errored')
            canceled = get_cnt(rc, 'canceled')
            expired = get_cnt(rc, 'expired')
            processing = get_cnt(rc, 'processing')
            
            total = succeeded + errored + canceled + expired + processing
            
            request_counts = RequestCounts(
                total=total,
                completed=succeeded,
                failed=errored
            )
        
        return BatchStatus(
            id=batch_result.id,
            status=status,
            completed=status == 'completed',
            failed=status == 'failed',
            output_file_id=None,
            error_file_id=None,
            results_url=getattr(batch_result, 'results_url', None),
            request_counts=request_counts
        )
    
    def download_results(
        self,
        batch_id: str,
        output_jsonl: str,
        force: bool = False
    ) -> bool:
        """Download batch results from Anthropic API if ready."""
        if Path(output_jsonl).exists() and not force:
            print(f"Results file already exists: {output_jsonl}")
            print("Use --force to re-download")
            return True
        
        print(f"Checking batch status: {batch_id}")
        status = self.check_batch_status(batch_id)
        print(f"Status: {status.status}")
        if status.request_counts:
            print(f"Progress: {status.request_counts.completed}/{status.request_counts.total} completed")
        
        # Allow download if we have a results_url, even if failed/canceled/expired
        if status.failed and not status.results_url:
            print("Error: Batch failed and no results found!")
            return False
        
        if not status.completed and not status.results_url:
            print("Batch not yet completed. Please wait and try again later.")
            return False
        
        print("Downloading results...")
        results = []
        
        try:
            for result in self.client.messages.batches.results(batch_id):
                # Convert Pydantic object to dict
                if hasattr(result, 'model_dump'):
                    results.append(result.model_dump())
                elif hasattr(result, 'dict'):
                    results.append(result.dict())
                else:
                    # Fallback for older Pydantic or unknowns
                    results.append(dict(result))
        except Exception as e:
            if status.results_url:
                import requests
                response = requests.get(status.results_url)
                if response.status_code == 200:
                    for line in response.text.splitlines():
                        if line.strip():
                            results.append(json.loads(line))
                else:
                    print(f"Error downloading from results_url: {response.status_code}")
                    return False
            else:
                print(f"Error downloading results: {e}")
                return False
        
        Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Downloaded {len(results)} records to {output_jsonl}")
        return True

    @staticmethod
    def parse_batch_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Anthropic batch result."""
        # result structure: {"custom_id": "...", "result": {"message": {...}}}
        
        custom_id = result.get('custom_id')
        
        # Check if successful
        if 'result' not in result or 'message' not in result['result']:
            # Maybe failed?
            return {
                'custom_id': custom_id,
                'raw_answer': '',
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'reasoning_tokens': 0,
                'reasoning_trace': '',
                'model': None,
                'error': 'No message in result'
            }
            
        message = result['result']['message']
        model = message.get('model')
        
        # Content
        content_list = message.get('content', [])
        texts = []
        reasoning_trace = None
        
        for block in content_list:
            if block.get('type') == 'text':
                texts.append(block.get('text', ''))
            elif block.get('type') == 'thinking':
                reasoning_trace = block.get('thinking', '')
                
        raw_answer = "".join(texts)
        
        # Usage
        usage = message.get('usage', {})
        prompt_tokens = usage.get('input_tokens', 0)
        completion_tokens = usage.get('output_tokens', 0)
        
        # Reasoning tokens
        reasoning_tokens = completion_tokens
        
        return {
            'custom_id': custom_id,
            'model': model,
            'raw_answer': raw_answer,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'reasoning_tokens': reasoning_tokens,
            'reasoning_trace': reasoning_trace if reasoning_trace else ''
        }

