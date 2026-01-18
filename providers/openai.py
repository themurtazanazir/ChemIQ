"""OpenAI provider client."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from openai import OpenAI

from .base import BaseThinkingLevel, ProviderClient, BatchStatus, RequestCounts, ModelConfig


class ThinkingLevel(BaseThinkingLevel):
    """OpenAI reasoning_effort values - in order of effort."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
    @property
    def color(self) -> str:
        """Pastel color for this thinking level."""
        colors = {
            "low": "#BFFCC6",     # green
            "medium": "#FFF7B2",  # yellow
            "high": "#FFB3BA",    # red
        }
        return colors.get(self.value, "#CCCCCC")


SUPPORTED_MODELS = {
    # Standard models
    "gpt-4o": ModelConfig(None, "#00838F"),      # Cyan 800
    "gpt-4o-mini": ModelConfig(None, "#0097A7"), # Cyan 700
    "gpt-4-turbo": ModelConfig(None, "#00ACC1"), # Cyan 600
    "gpt-4": ModelConfig(None, "#26C6DA"),       # Cyan 400
    "gpt-3.5-turbo": ModelConfig(None, "#4DD0E1"), # Cyan 300
    
    # Reasoning models
    "o1": ModelConfig(ThinkingLevel, "#6A1B9A"),        # Purple 800
    "o1-mini": ModelConfig(ThinkingLevel, "#7B1FA2"),   # Purple 700
    "o1-preview": ModelConfig(ThinkingLevel, "#8E24AA"),# Purple 600
    "o3-mini": ModelConfig(ThinkingLevel, "#AB47BC"),   # Purple 400
}


class OpenAIProviderClient(ProviderClient):
    """Provider client for OpenAI models."""
    
    ThinkingLevel = ThinkingLevel
    SUPPORTED_MODELS = SUPPORTED_MODELS
    ENV_VAR_NAME = "OPENAI_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
    
    def create_submission_file(
        self,
        questions: List[Dict[str, Any]],
        output_path: str,
        model_name: str,
        thinking: Union[ThinkingLevel, None] = None,
        **kwargs
    ) -> str:
        """Create a batch submission JSONL file for OpenAI models."""
        # Validate
        self.validate_thinking(model_name, thinking)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine thinking string for custom_id
        thinking_str = thinking.value if thinking else "0"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for question in questions:
                question_id = question["uuid"]
                prompt = question["prompt"]
                
                # Use strict UUID for custom_id to be safe and consistent
                custom_id = question_id
                
                body = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                
                # Add reasoning_effort for reasoning models
                if thinking:
                    body["reasoning_effort"] = thinking.value
                
                body.update(kwargs)
                
                record = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        return output_path
    
    def submit_batch(
        self,
        submission_file: str,
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> str:
        """Submit a batch job to OpenAI API."""
        with open(submission_file, 'rb') as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")
        
        batch_metadata = metadata or {}
        batch_metadata['description'] = submission_file
        
        created_batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=batch_metadata
        )
        
        return created_batch.id
    
    def check_batch_status(self, batch_id: str) -> BatchStatus:
        """Check the status of an OpenAI batch job."""
        batch_result = self.client.batches.retrieve(batch_id)
        
        request_counts = None
        if batch_result.request_counts:
            request_counts = RequestCounts(
                total=batch_result.request_counts.total,
                completed=batch_result.request_counts.completed,
                failed=batch_result.request_counts.failed
            )
        
        return BatchStatus(
            id=batch_result.id,
            status=batch_result.status,
            completed=batch_result.status == 'completed',
            failed=batch_result.status == 'failed',
            output_file_id=batch_result.output_file_id,
            error_file_id=batch_result.error_file_id,
            request_counts=request_counts
        )
    
    def download_results(
        self,
        batch_id: str,
        output_jsonl: str,
        force: bool = False
    ) -> bool:
        """Download batch results from OpenAI API if ready."""
        if Path(output_jsonl).exists() and not force:
            print(f"Results file already exists: {output_jsonl}")
            print("Use --force to re-download")
            return True
        
        print(f"Checking batch status: {batch_id}")
        status = self.check_batch_status(batch_id)
        print(f"Status: {status.status}")
        if status.request_counts:
            print(f"Progress: {status.request_counts.completed}/{status.request_counts.total} completed")
        
        if status.failed:
            print("Error: Batch failed!")
            if status.error_file_id:
                print(f"Error file ID: {status.error_file_id}")
            return False
        
        if not status.completed:
            print("Batch not yet completed. Please wait and try again later.")
            return False
        
        if not status.output_file_id:
            print("Error: No output file ID found")
            return False
        
        print("Downloading results...")
        output_file = self.client.files.content(status.output_file_id)
        data_str = output_file.read()
        if isinstance(data_str, bytes):
            data_str = data_str.decode('utf-8')
        results = [json.loads(line) for line in data_str.splitlines() if line.strip()]
        
        Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Downloaded {len(results)} records to {output_jsonl}")
        return True
    
    @staticmethod
    def parse_batch_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAI batch result."""
        # result structure: {"custom_id": "...", "response": {"body": {...}}}
        
        custom_id = result.get('custom_id')
        response = result.get('response', {})
        body = response.get('body', {})
        
        model = body.get('model')
        raw_answer = ''
        reasoning_trace = None
        
        choices = body.get('choices', [])
        if choices:
            msg = choices[0].get('message', {})
            raw_answer = msg.get('content', '')
            
            # Reasoning
            extra_content = msg.get('extraContent', {})
            if 'thoughtSignature' in extra_content:
                reasoning_trace = extra_content.get('thoughtSignature', '')
            elif 'reasoning_content' in msg:
                 # O1/O3
                reasoning_trace = msg.get('reasoning_content', '')
        
        usage = body.get('usage', {})
        prompt_tokens = usage.get('promptTokens', usage.get('prompt_tokens', 0))
        completion_tokens = usage.get('completionTokens', usage.get('completion_tokens', 0))
        
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

