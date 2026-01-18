
import os
import sys
from dotenv import load_dotenv

try:
    from anthropic import Anthropic
except ImportError:
    print("anthropic not installed")
    sys.exit(1)

load_dotenv(".env")
api_key = os.getenv("ANTHROPIC_API_KEY")

def cancel_batch(batch_id):
    client = Anthropic(api_key=api_key)
    try:
        print(f"Canceling batch {batch_id}...")
        client.messages.batches.cancel(batch_id)
        print("Cancel request sent.")
    except Exception as e:
        print(f"Error canceling batch: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python cancel_batch.py <batch_id>")
        sys.exit(1)
    cancel_batch(sys.argv[1])
