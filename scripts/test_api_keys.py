import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from src.utils.client_openrouter import OpenRouterClient
from src.utils.client_voyage import VoyageClient

def test():
    print("Configured OpenRouter keys:", [k[:10] + "..." for k in config.OPENROUTER_API_KEYS])
    print("Configured Model:", config.OPENROUTER_MODEL)
    print("Configured Voyage Key:", config.VOYAGE_API_KEY[:10] + "..." if config.VOYAGE_API_KEY else "NONE")

    print("\n1. Testing OpenRouter API call...")
    try:
        client = OpenRouterClient()
        res = client.generate(
            [{"role": "user", "content": 'Reply with JSON {"status": "ok", "message": "hello"}'}],
            json_mode=True,
            use_cache=False
        )
        print("OpenRouter SUCCESS! Response:\n", res)
    except Exception as e:
        print("OpenRouter EXCEPTION:", e)

    print("\n2. Testing Voyage AI API call...")
    try:
        vclient = VoyageClient()
        vecs = vclient.get_embeddings(["Hello world test"], use_cache=False)
        print("Voyage AI SUCCESS! Generated embedding vector length:", len(vecs[0]) if vecs else 0)
    except Exception as e:
        print("Voyage AI EXCEPTION:", e)

if __name__ == "__main__":
    test()
