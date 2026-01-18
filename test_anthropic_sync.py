
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

def test_single_request():
    client = Anthropic(api_key=api_key)
    
    print("Running single sync test request...")
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=20000,
            thinking={
                "type": "enabled",
                "budget_tokens": 16000
            },
            messages=[
                {"role": "user", "content": "What is 12 + 12? Think step by step."}
            ]
        )
        msg = f"Success!\nContent: {response.content}"
        print(msg)
        with open("test_output.txt", "w") as f:
            f.write(msg)
    except Exception as e:
        msg = f"Error: {e}"
        print(msg)
        with open("test_output.txt", "w") as f:
            f.write(msg)

if __name__ == "__main__":
    test_single_request()
