"""Google Gemini provider client."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from google import genai
from google.genai import types

from .base import BaseThinkingLevel, ProviderClient, BatchStatus, RequestCounts, ModelConfig


# Google Colors
C_FLASH = "#34A853"  # Green
C_PRO = "#4285F4"    # Blue
C_DEFAULT = "#EA4335" # Red


class ThinkingLevel(BaseThinkingLevel):
    """Google thinking_level values - in order of effort."""
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    
    @property
    def color(self) -> str:
        """Pastel color for this thinking level."""
        colors = {
            "MINIMAL": "#BAE1FF",  # blue
            "LOW": "#BFFCC6",      # green
            "MEDIUM": "#FFF7B2",   # yellow
            "HIGH": "#FFB3BA",     # red
        }
        return colors.get(self.value, "#CCCCCC")


SUPPORTED_MODELS = {
    # Standard models
    "gemini-1.5-pro": ModelConfig(None, "#1976D2"),     # Blue 700
    "gemini-1.5-flash": ModelConfig(None, "#388E3C"),   # Green 700
    "gemini-2.0-flash": ModelConfig(None, "#43A047"),   # Green 600
    
    # Thinking models with integer budget
    "gemini-2.0-flash-thinking-exp": ModelConfig("budget", "#4CAF50"), # Green 500
    "gemini-2.5-flash": ModelConfig("budget", "#66BB6A"), # Green 400
    "gemini-2.5-pro": ModelConfig("budget", "#42A5F5"),   # Blue 400
    
    # Thinking models with level enum
    "gemini-3-flash-preview": ModelConfig(ThinkingLevel, "#81C784"), # Green 300
    "gemini-3-pro-preview": ModelConfig(ThinkingLevel, "#64B5F6"),   # Blue 300
}


class GoogleProviderClient(ProviderClient):
    """Provider client for Google Gemini models."""
    
    ThinkingLevel = ThinkingLevel
    SUPPORTED_MODELS = SUPPORTED_MODELS
    ENV_VAR_NAME = "GOOGLE_API_KEY"
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = genai.Client(api_key=api_key)
    
    def create_submission_file(
        self,
        questions: List[Dict[str, Any]],
        output_path: str,
        model_name: str,
        thinking: Union[ThinkingLevel, int, None] = None,
        **kwargs
    ) -> str:
        """Create a batch submission JSONL file for Google models."""
        # Validate
        self.validate_thinking(model_name, thinking)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Determine thinking string for custom_id
        if thinking is None:
            thinking_str = "0"
        elif isinstance(thinking, ThinkingLevel):
            thinking_str = thinking.name.lower()  # "minimal", "low", etc.
        else:
            thinking_str = str(thinking)  # integer budget
        
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
                
                # Add thinking_config if thinking specified
                if thinking:
                    thinking_config = {"include_thoughts": True}
                    
                    if isinstance(thinking, ThinkingLevel):
                        thinking_config["thinking_level"] = thinking.value
                    elif isinstance(thinking, int):
                        thinking_config["thinking_budget"] = thinking
                    
                    body["extra_body"] = {
                        "google": {"thinking_config": thinking_config}
                    }
                
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
        """Submit a batch job to Google API."""
        # Upload file
        batch_input_file = self.client.files.upload(
            file=submission_file,
            config=types.UploadFileConfig(
                display_name='batch-requests',
                mime_type='application/jsonl'
            )
        )
        
        file_id = batch_input_file.name
        if not file_id.startswith("files/"):
            file_id = f"files/{file_id}"
        
        # Extract model name from submission file
        model_name = None
        with open(submission_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line.strip():
                first_record = json.loads(first_line)
                model_name = first_record.get('body', {}).get('model')
        
        if not model_name:
            raise ValueError("Could not determine model name from submission file")
        
        # Create batch
        batch_config = types.CreateBatchJobConfig()
        
        if metadata:
            safe_labels = {}
            for k, v in metadata.items():
                safe_k = str(k).lower().replace(' ', '_').replace('.', '_')
                safe_v = str(v).lower().replace(' ', '_')
                safe_labels[safe_k] = safe_v
            try:
                batch_config.labels = safe_labels
            except (AttributeError, ValueError):
                pass
        
        created_batch = self.client.batches.create(
            model=model_name,
            src=file_id,
            config=batch_config
        )
        
        return created_batch.name
    
    def check_batch_status(self, batch_id: str) -> BatchStatus:
        """Check the status of a Google batch job."""
        batch_result = self.client.batches.get(name=batch_id)
        
        state = batch_result.state
        state_str = state.name if hasattr(state, 'name') else str(state)
        
        status_map = {
            'JOB_STATE_SUCCEEDED': 'completed',
            'JOB_STATE_FAILED': 'failed',
            'JOB_STATE_RUNNING': 'in_progress',
            'JOB_STATE_PENDING': 'pending',
            'JOB_STATE_QUEUED': 'pending',
        }
        status = status_map.get(state_str, state_str.lower().replace('job_state_', ''))
        
        request_counts = None
        completion_stats = batch_result.completion_stats
        if completion_stats:
            total = (completion_stats.successful_count or 0) + \
                   (completion_stats.failed_count or 0) + \
                   (completion_stats.incomplete_count or 0)
            request_counts = RequestCounts(
                total=total,
                completed=completion_stats.successful_count or 0,
                failed=completion_stats.failed_count or 0
            )
        
        output_file_id = None
        if batch_result.dest:
            if batch_result.dest.file_name:
                output_file_id = batch_result.dest.file_name
            elif batch_result.dest.gcs_uri:
                output_file_id = batch_result.dest.gcs_uri
        
        metadata = None
        if hasattr(batch_result, 'labels') and batch_result.labels:
            metadata = dict(batch_result.labels)

        return BatchStatus(
            id=batch_result.name,
            status=status,
            completed=state_str == 'JOB_STATE_SUCCEEDED',
            failed=state_str == 'JOB_STATE_FAILED',
            output_file_id=output_file_id,
            error_file_id=None,
            request_counts=request_counts,
            metadata=metadata
        )
    
    def download_results(
        self,
        batch_id: str,
        output_jsonl: str,
        force: bool = False
    ) -> bool:
        """Download batch results from Google API if ready."""
        if Path(output_jsonl).exists() and not force:
            print(f"Results file already exists: {output_jsonl}")
            print("Use --force to re-download")
            return True
        
        print(f"Checking batch status: {batch_id}")
        status = self.check_batch_status(batch_id)
        
        batch_result = self.client.batches.get(name=batch_id)
        print(f"Status: {status.status}")
        print(f"Model: {batch_result.model}")
        if batch_result.create_time:
            print(f"Created: {batch_result.create_time}")
        if status.request_counts:
            print(f"Progress: {status.request_counts.completed}/{status.request_counts.total} completed")
        
        if status.failed:
            print("Error: Batch failed!")
            return False
        
        if not status.completed:
            print("Batch not yet completed. Please wait and try again later.")
            return False
        
        print("Downloading results...")
        
        if batch_result.dest and batch_result.dest.file_name:
            output_file_response = self.client.files.download(file=batch_result.dest.file_name)
            data_str = output_file_response.decode('utf-8')
            results = [json.loads(line) for line in data_str.splitlines() if line.strip()]
        elif batch_result.dest and batch_result.dest.inlined_responses:
            results = []
            for response in batch_result.dest.inlined_responses:
                results.append({
                    'custom_id': getattr(response, 'custom_id', None),
                    'response': {
                        'body': response.model_dump() if hasattr(response, 'model_dump') else dict(response)
                    }
                })
        elif batch_result.dest and batch_result.dest.gcs_uri:
            print(f"Error: Results stored in GCS: {batch_result.dest.gcs_uri}")
            print("GCS download not yet implemented.")
            return False
        else:
            print("Error: No output destination found")
            return False
        
        Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Downloaded {len(results)} records to {output_jsonl}")
        return True

    @staticmethod
    def parse_batch_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Google batch result."""
        # result structure: {"custom_id": "...", "response": {"body": {...}}}
        
        custom_id = result.get('custom_id')
        response = result.get('response', {})
        body = response.get('body', {})
        
        # GoogleGenAI response structure
        model = body.get('model') # Might not be present in body, but let's check
        raw_answer = ''
        
        candidates = body.get('candidates', [])
        if candidates:
            content = candidates[0].get('content', {})
            parts = content.get('parts', [])
            texts = []
            for part in parts:
                if 'text' in part:
                    texts.append(part['text'])
            raw_answer = "".join(texts)
        
        # OpenAI-compatible format (since we use /v1/chat/completions endpoint)
        if not raw_answer:
            choices = body.get('choices', [])
            if choices:
                message = choices[0].get('message', {})
                raw_answer = message.get('content', '')
            
        usage = body.get('usage_metadata', {})
        prompt_tokens = usage.get('prompt_token_count', 0)
        completion_tokens = usage.get('candidates_token_count', 0)
        
        # OpenAI-compatible usage format
        if prompt_tokens == 0 and completion_tokens == 0:
            usage = body.get('usage', {})
            prompt_tokens = usage.get('promptTokens', 0) or usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completionTokens', 0) or usage.get('completion_tokens', 0)
        
        # Reasoning trace (Thinking config)
        # Google thinking is usually in text or parts? 
        # For now assume mixed in text or separate part? 
        # Current logic doesn't extract it separately for Google yet.
        reasoning_trace = None
        
        # Reasoning tokens same as completion for now
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

