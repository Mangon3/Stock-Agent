import os
import sys
import time
from src.config.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI

print("--- DIAGNOSTIC START ---")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Python Path: {sys.path}")

print(f"Checking GOOGLE_API_KEY: {'Present' if settings.GOOGLE_API_KEY else 'Missing'}")
if settings.GOOGLE_API_KEY:
    print(f"Key length: {len(settings.GOOGLE_API_KEY)}")
    # Print first/last chars for sanity check (safe-ish)
    print(f"Key hint: {settings.GOOGLE_API_KEY[:4]}...{settings.GOOGLE_API_KEY[-4:]}")

print(f"Checking FINNHUB_API_KEY: {'Present' if settings.FINNHUB_API_KEY else 'Missing'}")

print("Testing LLM connectivity (Google Gemini)...")
try:
    start_time = time.time()
    llm = ChatGoogleGenerativeAI(
        model=settings.MODEL, 
        api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,
        max_retries=1,
        request_timeout=10 # Short timeout for test
    )
    res = llm.invoke("Hello, output the word 'Success'.")
    end_time = time.time()
    print(f"LLM Response: {res.content}")
    print(f"LLM Latency: {end_time - start_time:.2f}s")
except Exception as e:
    print(f"LLM Failed: {e}")
    # Print traceback if needed
    import traceback
    traceback.print_exc()

print("Testing Tool execution (Micro Model check)...")
try:
    from src.tools.micro import micro_model
    print("MicroModel imported successfully.")
    # We won't run full training, just check if method exists
    if hasattr(micro_model, 'execute_model_training'):
        print("execute_model_training method found.")
except Exception as e:
    print(f"Tool check failed: {e}")
    import traceback
    traceback.print_exc()

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: GPU not available. Check NVIDIA drivers and container runtime.")

print("--- DIAGNOSTIC END ---")
