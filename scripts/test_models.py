"""
Test script to verify which Helsinki-NLP models are available
"""

from transformers import MarianMTModel, MarianTokenizer

MODELS_TO_TEST = [
    "Helsinki-NLP/opus-tatoeba-en-tr",
    "Helsinki-NLP/opus-mt-tc-big-en-tr",
    "Helsinki-NLP/opus-mt-tc-big-tr-en",
    "Helsinki-NLP/opus-mt-tr-en",
]

print("\n" + "="*60)
print("Testing Model Availability")
print("="*60 + "\n")

available_models = []

for model_name in MODELS_TO_TEST:
    try:
        print(f"Testing: {model_name}")
        # Try to load just the config (fast test)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        print(f"  ✓ Available! (Config loaded)")
        available_models.append(model_name)
    except Exception as e:
        error_msg = str(e)[:80]
        print(f"  ✗ Not available: {error_msg}...")

print("\n" + "="*60)
print(f"Summary: {len(available_models)}/{len(MODELS_TO_TEST)} models available")
print("="*60)

if available_models:
    print("\n✓ Available models:")
    for model in available_models:
        print(f"  - {model}")
    print(f"\nRecommendation: Use '{available_models[0]}' (first available)")
else:
    print("\n✗ No models available. Check your internet connection.")
