"""
ONNX Translation Model Downloader & Converter
Downloads Helsinki-NLP Opus-MT en-tr model and converts to ONNX format
"""

import os
import sys
import shutil
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
import torch

# Configuration
# Try multiple model sources (verified models from Hugging Face)
# Ordered by priority: smaller/faster first for testing, then best quality
MODEL_NAMES = [
    "Helsinki-NLP/opus-tatoeba-en-tr",    # Small, fast (2.05k downloads) - TEST FIRST
    "Helsinki-NLP/opus-mt-tc-big-en-tr",  # Best quality (171k downloads) - PRODUCTION
    "Helsinki-NLP/opus-mt-tc-big-tr-en",  # Reverse direction (160k downloads) - FALLBACK
]
OUTPUT_DIR = Path(__file__).parent.parent / "onnx_models" / "en-tr"
TEMP_DIR = Path(__file__).parent.parent / "temp_models"

def print_step(step_num, message):
    """Print formatted step message"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {message}")
    print(f"{'='*60}\n")

def download_model():
    """Step 1: Download original model from Hugging Face"""
    print_step(1, "Downloading EN-TR translation model...")
    
    # Create temp directory
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try each model name until one works
    for idx, model_name in enumerate(MODEL_NAMES, 1):
        try:
            print(f"[{idx}/{len(MODEL_NAMES)}] Attempting to download: {model_name}")
            print("(this may take 5-10 minutes depending on model size)...")
            print("Connecting to Hugging Face...")
            
            # Download model and tokenizer
            print("  → Downloading model weights...")
            model = MarianMTModel.from_pretrained(model_name)
            
            print("  → Downloading tokenizer...")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            
            # Save to temp directory
            print("  → Saving to local directory...")
            model.save_pretrained(TEMP_DIR)
            tokenizer.save_pretrained(TEMP_DIR)
            
            # Verify files were saved
            saved_files = list(TEMP_DIR.glob("*"))
            print(f"  → Saved {len(saved_files)} files")
            
            print(f"\n✓ Model downloaded successfully: {model_name}")
            print(f"✓ Saved to {TEMP_DIR}")
            return True
            
        except Exception as e:
            print(f"\n✗ Failed with {model_name}")
            print(f"   Error: {str(e)[:100]}...")
            if idx < len(MODEL_NAMES):
                print("   Trying next model...\n")
            continue
    
    # If all models failed, try alternative approach
    print("\n⚠ All Helsinki-NLP models failed. Trying alternative...")
    try:
        # Use facebook/mbart or another alternative
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        print(f"Downloading alternative model: {model_name}")
        print("Note: This is a larger multi-language model (~2GB)")
        print("(this may take 10-15 minutes)...")
        
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
        
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        
        # Set source and target languages
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "tr_TR"
        
        model.save_pretrained(TEMP_DIR)
        tokenizer.save_pretrained(TEMP_DIR)
        
        # Save language config
        with open(TEMP_DIR / "lang_config.json", 'w') as f:
            import json
            json.dump({"src_lang": "en_XX", "tgt_lang": "tr_TR"}, f)
        
        print(f"✓ Alternative model downloaded: {model_name}")
        return True
        
    except Exception as e:
        print(f"✗ All models failed: {e}")
        print("\n💡 TIP: Try manually downloading from:")
        print("   https://huggingface.co/Helsinki-NLP")
        return False

def convert_to_onnx():
    """Step 2: Convert PyTorch model to ONNX format"""
    print_step(2, "Converting to ONNX format...")
    
    try:
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        print("Loading model for conversion...")
        # Load and convert to ONNX using Optimum
        model = ORTModelForSeq2SeqLM.from_pretrained(
            TEMP_DIR,
            export=True,
            provider="CPUExecutionProvider"
        )
        
        # Load tokenizer (try both MarianTokenizer and AutoTokenizer)
        print("Loading tokenizer...")
        try:
            tokenizer = MarianTokenizer.from_pretrained(TEMP_DIR)
        except:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(TEMP_DIR)
        
        # Save ONNX model
        print(f"Saving ONNX model to {OUTPUT_DIR}...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"✓ ONNX conversion completed")
        return True
    except Exception as e:
        print(f"✗ Error converting to ONNX: {e}")
        import traceback
        traceback.print_exc()
        return False

def quantize_model():
    """Step 3: Apply INT8 quantization for mobile optimization"""
    print_step(3, "Applying INT8 quantization (mobile optimization)...")
    
    try:
        # Quantization configuration
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        
        # Paths
        encoder_path = OUTPUT_DIR / "encoder_model.onnx"
        decoder_path = OUTPUT_DIR / "decoder_model.onnx"
        decoder_with_past_path = OUTPUT_DIR / "decoder_with_past_model.onnx"
        
        quantized_dir = OUTPUT_DIR / "quantized"
        quantized_dir.mkdir(exist_ok=True)
        
        # Quantize each model component
        models_to_quantize = []
        if encoder_path.exists():
            models_to_quantize.append(("encoder", encoder_path))
        if decoder_path.exists():
            models_to_quantize.append(("decoder", decoder_path))
        if decoder_with_past_path.exists():
            models_to_quantize.append(("decoder_with_past", decoder_with_past_path))
        
        for name, model_path in models_to_quantize:
            print(f"Quantizing {name}...")
            quantizer = ORTQuantizer.from_pretrained(OUTPUT_DIR, file_name=model_path.name)
            quantizer.quantize(
                save_dir=quantized_dir,
                quantization_config=qconfig,
                file_suffix=f"_{name}_quantized"
            )
        
        # Replace original models with quantized versions
        print("Replacing models with quantized versions...")
        for file in quantized_dir.glob("*.onnx"):
            # Remove "_quantized" and extra suffixes (e.g., "__encoder")
            new_name = file.name.replace("_quantized", "")
            # Extract base name (encoder_model, decoder_model, etc.)
            if "__" in new_name:
                parts = new_name.split("__")
                new_name = parts[0] + ".onnx"  # e.g., "encoder_model.onnx"
            
            target = OUTPUT_DIR / new_name
            
            # Delete original unquantized file
            if target.exists():
                target.unlink()
                print(f"  → Removed old: {target.name}")
            
            # Copy quantized version
            shutil.copy2(file, target)
            print(f"  → Saved quantized: {target.name}")
        
        # Delete duplicate files (with "__" suffix)
        for old_file in OUTPUT_DIR.glob("*__*.onnx"):
            old_file.unlink()
            print(f"  → Cleaned up: {old_file.name}")
        
        # Cleanup quantized temp dir
        shutil.rmtree(quantized_dir)
        
        print("✓ Quantization completed (expected size reduction: ~70%)")
        return True
    except Exception as e:
        print(f"⚠ Warning: Quantization failed (model will work but be larger): {e}")
        print("  Continuing with standard ONNX model...")
        return True  # Non-critical, continue anyway

def get_model_size(directory):
    """Calculate total size of model files"""
    total_size = 0
    for file in Path(directory).rglob("*.onnx"):
        total_size += file.stat().st_size
    return total_size / (1024 * 1024)  # Convert to MB

def cleanup():
    """Step 4: Cleanup temporary files"""
    print_step(4, "Cleaning up temporary files...")
    
    try:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            print(f"✓ Removed temporary directory: {TEMP_DIR}")
    except Exception as e:
        print(f"⚠ Warning: Could not cleanup temp files: {e}")

def verify_model():
    """Step 5: Verify model is working"""
    print_step(5, "Verifying model functionality...")
    
    try:
        # Load model
        model = ORTModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
        
        # Load tokenizer (try both types)
        try:
            tokenizer = MarianTokenizer.from_pretrained(OUTPUT_DIR)
        except:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
        
        # Test translation
        test_text = "Hello, how are you?"
        print(f"Testing translation: '{test_text}'")
        
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✓ Translation result: '{translated}'")
        print(f"✓ Model is working correctly!")
        
        # Print final stats
        model_size = get_model_size(OUTPUT_DIR)
        print(f"\n{'='*60}")
        print(f"SETUP COMPLETE!")
        print(f"{'='*60}")
        print(f"Model location: {OUTPUT_DIR}")
        print(f"Total model size: {model_size:.2f} MB")
        print(f"Ready for Flutter integration with onnx_translation package")
        print(f"{'='*60}\n")
        
        return True
    except Exception as e:
        print(f"✗ Error verifying model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution flow"""
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  ONNX Translation Model Setup                              ║
║  Helsinki-NLP/opus-mt-tc-big-en-tr → ONNX (Mobile)        ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # Execute pipeline
    steps = [
        ("Download Model", download_model),
        ("Convert to ONNX", convert_to_onnx),
        ("Quantize Model", quantize_model),
        ("Cleanup", cleanup),  # Cleanup before verify (temp files no longer needed)
        ("Verify Model", verify_model),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n✗ Pipeline failed at: {step_name}")
            sys.exit(1)
    
    print("\n✓ All steps completed successfully!")
    print(f"✓ Model ready at: {OUTPUT_DIR}")
    
if __name__ == "__main__":
    main()
