
import asyncio
import os
from dotenv import load_dotenv
from src.soul.llm_client import LLMClient

load_dotenv()

async def test_llm():
    print("Testing LLMClient...")
    
    # Check key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found in env.")
        return

    client = LLMClient(model="gemini-2.0-flash")
    print("Client created.")
    
    try:
        print("Sending generation request...")
        response = await client.generate_content("Say hello.")
        
        if hasattr(response, 'content'):
            print(f"Response: {response.content}")
            if hasattr(response, 'usage'):
                print(f"Usage: {response.usage}")
        else:
            print(f"Response (str): {response}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_llm())
