"""
Test All Translation Models
Verifies that all language pair models are working correctly
Tests each model with sample translations
"""

import sys
from pathlib import Path
from transformers import MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import json
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "onnx_models"

# Test sentences for each language
TEST_SENTENCES = {
    'en': 'Hello, how are you?',
    'tr': 'Merhaba, nasılsın?',
    'de': 'Hallo, wie geht es dir?',
    'fr': 'Bonjour, comment allez-vous?',
    'it': 'Ciao, come stai?',
    'pt': 'Olá, como você está?',
    'es': 'Hola, ¿cómo estás?',
}

def print_header():
    """Print script header"""
    print("""
╔════════════════════════════════════════════════════════════╗
║  Translation Models Tester                                ║
║  Testing all 42 language pairs                           ║
╚════════════════════════════════════════════════════════════╝
""")

def test_translation(pair_dir, source_lang, target_lang):
    """
    Test a single language pair
    Returns: (success, translation, error_message)
    """
    try:
        # Load model
        model = ORTModelForSeq2SeqLM.from_pretrained(pair_dir)
        
        # Load tokenizer
        try:
            tokenizer = MarianTokenizer.from_pretrained(pair_dir)
        except:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(pair_dir)
        
        # Get test sentence
        test_text = TEST_SENTENCES.get(source_lang, "Hello world")
        
        # Translate
        inputs = tokenizer(test_text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs, max_length=100)
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return True, translation, None
        
    except Exception as e:
        error_msg = str(e)[:100]
        return False, None, error_msg

def get_model_info(pair_dir):
    """Get model metadata"""
    metadata_file = pair_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def test_all_models():
    """Test all language pair models"""
    print("\nScanning for models...\n")
    
    # Find all language pair directories
    pair_dirs = sorted([d for d in MODELS_DIR.glob("*-*") if d.is_dir()])
    
    if not pair_dirs:
        print("✗ No model directories found!")
        print(f"  Expected location: {MODELS_DIR}")
        return
    
    print(f"Found {len(pair_dirs)} language pairs to test\n")
    print("="*60)
    
    results = {
        'passed': [],
        'failed': [],
        'total': len(pair_dirs)
    }
    
    for i, pair_dir in enumerate(pair_dirs, 1):
        pair_name = pair_dir.name
        source_lang, target_lang = pair_name.split('-')
        
        print(f"\n[{i}/{len(pair_dirs)}] Testing: {pair_name}")
        
        # Get model info
        info = get_model_info(pair_dir)
        size = info.get('size_mb', 0)
        
        # Test translation
        success, translation, error = test_translation(pair_dir, source_lang, target_lang)
        
        if success:
            test_text = TEST_SENTENCES.get(source_lang, "Hello world")
            print(f"  Input:  '{test_text}'")
            print(f"  Output: '{translation}'")
            print(f"  Size:   {size:.1f} MB")
            print(f"  ✓ PASSED")
            results['passed'].append({
                'pair': pair_name,
                'size_mb': size,
                'translation': translation
            })
        else:
            print(f"  ✗ FAILED: {error}")
            results['failed'].append({
                'pair': pair_name,
                'error': error
            })
    
    # Save test results
    test_results_file = BASE_DIR / "test_results.json"
    results['tested_at'] = datetime.now().isoformat()
    
    with open(test_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = len(results['passed'])
    failed = len(results['failed'])
    total = results['total']
    
    print(f"\n✓ Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"✗ Failed: {failed}/{total}")
    
    if results['failed']:
        print(f"\nFailed models:")
        for item in results['failed']:
            print(f"  - {item['pair']}: {item['error'][:60]}...")
    
    # Calculate total size
    total_size = sum(r['size_mb'] for r in results['passed'])
    avg_size = total_size / passed if passed > 0 else 0
    
    print(f"\nTotal size (working models): {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"Average model size: {avg_size:.1f} MB")
    
    print(f"\nTest results saved to: {test_results_file}")
    print("="*60 + "\n")
    
    return passed == total

def main():
    """Main execution"""
    print_header()
    
    if not MODELS_DIR.exists():
        print("✗ Error: onnx_models directory not found!")
        print("  Run download_all_languages.py first.")
        sys.exit(1)
    
    all_passed = test_all_models()
    
    if all_passed:
        print("✓ All models are working correctly!")
        print("\nReady for deployment!")
        print("Next step: Upload to Hugging Face for CDN distribution\n")
        sys.exit(0)
    else:
        print("⚠ Some models failed testing.")
        print("  Review the failed models and re-run download for those pairs.\n")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Testing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
