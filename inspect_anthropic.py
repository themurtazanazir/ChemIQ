
import os
import sys
from dotenv import load_dotenv

# Mocking ProviderClient to avoid huge imports
class ProviderClient:
    def __init__(self, api_key):
        self.api_key = api_key

try:
    from anthropic import Anthropic
except ImportError:
    print("anthropic not installed")
    sys.exit(1)

load_dotenv(".env")
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("No ANTHROPIC_API_KEY found")
    # sys.exit(1) # Don't exit, might be passed, but here we just need to test import/structure if possible
    # We need a valid key to actually hit the API. 
    # But wait, looking at the error traceback, the error happened inside check_batch_status -> retrieve.
    # The retrieve succeeded! It returned a MessageBatch object.
    # Then we accessed .status and it failed.
    pass

def check_structure():
    # We can't easily reproduce the object without hitting the API with a valid ID.
    # The ID in the traceback is msgbatch_01XNhUiMMQo7WFUuRsWQ3fnK
    
    client = Anthropic(api_key=api_key)
    try:
        batch = client.messages.batches.retrieve("msgbatch_01XNhUiMMQo7WFUuRsWQ3fnK")
        print("\n--- Batch Details ---")
        print(f"ID: {batch.id}")
        print(f"Processing Status: {batch.processing_status}")
        print(f"Request Counts: {batch.request_counts}")
        print(f"Created At: {batch.created_at}")
        print(f"Expires At: {batch.expires_at}")
        print(f"Errors: {getattr(batch, 'errors', 'No top-level errors')}")
        
        if batch.processing_status == "in_progress":
            print("\nWait diagnosis: Large batches with reasoning/thinking can take significant time.")
            print("But 0/816 after hours is suspicious.")
            
    except Exception as e:
        print(f"Error checking batch: {e}")

if __name__ == "__main__":
    check_structure()
