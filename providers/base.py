"""Base classes for provider clients."""

from abc import ABC, abstractmethod
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


class BaseThinkingLevel(Enum):
    """Base class for thinking level enums with ordering support."""
    
    @property
    def order(self) -> int:
        """Order by enum definition order (0-indexed)."""
        return list(self.__class__).index(self)
    
    def __lt__(self, other):
        if not isinstance(other, BaseThinkingLevel):
            return NotImplemented
        return self.order < other.order
    
    def __le__(self, other):
        if not isinstance(other, BaseThinkingLevel):
            return NotImplemented
        return self.order <= other.order
    
    def __gt__(self, other):
        if not isinstance(other, BaseThinkingLevel):
            return NotImplemented
        return self.order > other.order
    
    def __ge__(self, other):
        if not isinstance(other, BaseThinkingLevel):
            return NotImplemented
        return self.order >= other.order


@dataclass
class RequestCounts:
    """Request count statistics for a batch job."""
    total: int
    completed: int
    failed: int


@dataclass
class BatchStatus:
    """Status information for a batch job."""
    id: str
    status: str
    completed: bool
    failed: bool
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    request_counts: Optional[RequestCounts] = None
    results_url: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class ModelConfig:
    """Configuration for a supported model."""
    thinking_mode: Union[type, str, None]  # enum type, "budget", or None
    color: str = "#000000"


class ProviderClient(ABC):
    """Abstract base class for LLM provider clients."""
    
    # Subclasses should define these
    ThinkingLevel = None  # The thinking level enum for this provider
    SUPPORTED_MODELS: Dict[str, Union[ModelConfig, Any]] = {}  # Model -> config
    ENV_VAR_NAME: str = ""  # Environment variable for API key

    
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    @classmethod
    def validate_submission(cls, api_key: Optional[str], model_name: str, thinking=None) -> str:
        """Validate all submission inputs."""
        # Validate API Key
        valid_key = api_key or os.environ.get(cls.ENV_VAR_NAME)
        if not valid_key:
            raise ValueError(f"API key required. Set {cls.ENV_VAR_NAME} environment variable or pass explicitly.")
            
        # Validate Thinking/Model
        if thinking is None:
            return cls(api_key=valid_key)
            
        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
            
        raw_config = cls.SUPPORTED_MODELS[model_name]
        
        # Normalize config to thinking mode
        thinking_mode = raw_config
        if isinstance(raw_config, ModelConfig):
            thinking_mode = raw_config.thinking_mode
            
        if thinking_mode is None:
            raise ValueError(f"Model '{model_name}' does not support thinking/reasoning")
        
        if thinking_mode == "budget":
            if not isinstance(thinking, int):
                raise ValueError(f"Model '{model_name}' requires integer thinking_budget, got {type(thinking).__name__}")
            return cls(api_key=valid_key)
            
        if isinstance(thinking_mode, type) and issubclass(thinking_mode, BaseThinkingLevel):
            if not isinstance(thinking, thinking_mode):
                valid = [level.name for level in thinking_mode]
                f_val = thinking
                raise ValueError(f"Model '{model_name}' supports {valid}, got '{f_val}'")
        
        return cls(api_key=valid_key)

    @classmethod
    def validate_download(cls, api_key: Optional[str], batch_id: str) -> str:
        """Validate all download inputs.
        
        Args:
            api_key: The API key
            batch_id: The batch ID
            
        Returns:
            The resolved API key
            
        Raises:
            ValueError: If invalid
        """
        valid_key = api_key or os.environ.get(cls.ENV_VAR_NAME)
        if not valid_key:
            raise ValueError(f"API key required. Set {cls.ENV_VAR_NAME} environment variable or pass explicitly.")
            
        if not batch_id:
            raise ValueError("Batch ID is required")
            
        return cls(api_key=valid_key)
            
    def validate_thinking(self, model_name: str, thinking=None):
        """Instance alias for validation."""
        self.validate_submission(self.api_key, model_name, thinking)
    
    @abstractmethod
    def create_submission_file(
        self,
        questions: List[Dict[str, Any]],
        output_path: str,
        model_name: str,
        thinking: Union['BaseThinkingLevel', int, None] = None,
        **kwargs
    ) -> str:
        """Create a batch submission file."""
        pass
    
    @abstractmethod
    def submit_batch(
        self,
        submission_file: str,
        completion_window: str = "24h",
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Submit a batch job. Returns batch ID."""
        pass
    
    @abstractmethod
    def check_batch_status(self, batch_id: str) -> BatchStatus:
        """Check the status of a batch job."""
        pass
    
    @abstractmethod
    def download_results(
        self,
        batch_id: str,
        output_jsonl: str,
        force: bool = False
    ) -> bool:
        """Download batch results if ready."""
        pass

    @staticmethod
    @abstractmethod
    def parse_batch_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single batch result line into a standard dict.
        
        Returns dict with:
            - custom_id
            - model (optional)
            - raw_answer
            - prompt_tokens
            - completion_tokens
            - reasoning_tokens
            - reasoning_trace
        """
        pass
    
    def get_batch_id_filename(self, model_name: str, thinking=None) -> str:
        """Generate a filename for storing batch ID."""
        safe_model_name = model_name.replace("/", "-").replace("_", "-")
        
        if thinking is None:
            return f"batch_ids/{safe_model_name}.txt"
        
        # Handle enum or int
        if isinstance(thinking, BaseThinkingLevel):
            suffix = thinking.name.lower()
        else:
            suffix = str(thinking)
        
        return f"batch_ids/{safe_model_name}-tb{suffix}.txt"
    
    def save_batch_id(self, batch_id: str, model_name: str, thinking=None) -> str:
        """Save batch ID to a file."""
        filename = self.get_batch_id_filename(model_name, thinking)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            f.write(batch_id)
        return filename


class SyncProviderClient(ABC):
    """Abstract base class for synchronous (real-time) LLM providers.
    
    Unlike ProviderClient (batch APIs), sync providers query models
    in real-time and write results directly to CSV.
    """
    
    SUPPORTED_MODELS: Dict[str, ModelConfig] = {}
    
    @abstractmethod
    def query(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        """Query the model synchronously.
        
        Returns:
            Dict with keys: raw_answer, prompt_tokens, completion_tokens
        """
        pass
    
    def get_model_name(self) -> str:
        """Return the model identifier for CSV output."""
        return list(self.SUPPORTED_MODELS.keys())[0] if self.SUPPORTED_MODELS else "unknown"
    
    def _process_question(self, question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single question and return the result row."""
        model_name = self.get_model_name()
        try:
            result = self.query(question["prompt"])
            return {
                "model": model_name,
                "uuid": question["uuid"],
                "thinking_budget": 0,
                "prompt_tokens": result.get("prompt_tokens", 0),
                "reasoning_trace": "",
                "reasoning_tokens": 0,
                "raw_model_answer": result.get("raw_answer", ""),
                "total_tokens": result.get("prompt_tokens", 0) + result.get("completion_tokens", 0),
            }
        except Exception as e:
            print(f"Error on {question['uuid']}: {e}")
            return None
    
    def run_evaluation(
        self,
        questions: List[Dict[str, Any]],
        output_csv: str,
        resume: bool = True,
        max_workers: int = 1,
    ) -> None:
        """Run evaluation and write results to CSV.
        
        Args:
            questions: List of question dicts with 'uuid' and 'prompt' keys
            output_csv: Path to output CSV file
            resume: If True, skip already-processed UUIDs
            max_workers: Number of parallel threads (default: 1 for sequential)
        """
        import csv
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        model_name = self.get_model_name()
        fieldnames = [
            "model", "uuid", "thinking_budget", "prompt_tokens",
            "reasoning_trace", "reasoning_tokens", "raw_model_answer", "total_tokens"
        ]
        
        # Load existing UUIDs if resuming
        processed_uuids = set()
        if resume and Path(output_csv).exists():
            with open(output_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_uuids.add(row.get("uuid"))
            print(f"Resuming: {len(processed_uuids)} already processed")
        
        # Filter questions
        remaining = [q for q in questions if q["uuid"] not in processed_uuids]
        print(f"Processing {len(remaining)} questions with {max_workers} worker(s)...")
        
        # Open CSV for appending
        file_exists = Path(output_csv).exists() and resume
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe file writing
        write_lock = threading.Lock()
        
        with open(output_csv, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            if max_workers == 1:
                # Sequential processing (original behavior)
                for q in tqdm(remaining, desc=f"Evaluating {model_name}"):
                    row = self._process_question(q)
                    if row:
                        writer.writerow(row)
                        f.flush()
            else:
                # Parallel processing with thread pool
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(self._process_question, q): q for q in remaining}
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {model_name}"):
                        row = future.result()
                        if row:
                            with write_lock:
                                writer.writerow(row)
                                f.flush()
        
        print(f"Results written to {output_csv}")

