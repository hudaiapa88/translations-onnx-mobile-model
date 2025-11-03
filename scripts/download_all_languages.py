"""
Multi-Language ONNX Translation Model Downloader & Converter
Downloads all language pairs and converts to optimized ONNX format
Automatically cleans up PyTorch models after conversion to save disk space
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
from transformers import MarianMTModel, MarianTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
import torch

# Import language configuration
from language_config import (
    ALL_LANGUAGE_PAIRS, 
    get_language_pair_info,
    LANGUAGE_NAMES
)

# Directories
BASE_DIR = Path(__file__).parent.parent
OUTPUT_BASE_DIR = BASE_DIR / "onnx_models"
TEMP_DIR = BASE_DIR / "temp_downloads"
LOG_FILE = BASE_DIR / "download_log.json"

# Progress tracking
progress_data = {
    "started_at": None,
    "total_pairs": len(ALL_LANGUAGE_PAIRS),
    "completed": [],
    "failed": [],
    "skipped": [],
    "current": None
}

def load_progress():
    """Load existing progress from log file"""
    global progress_data
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            progress_data.update(json.load(f))
        print(f"\n✓ Loaded previous progress: {len(progress_data['completed'])} completed, {len(progress_data['failed'])} failed")

def save_progress():
    """Save current progress to log file"""
    with open(LOG_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)

def print_header():
    """Print script header"""
    print("""
╔════════════════════════════════════════════════════════════╗
║  Multi-Language ONNX Translation Model Setup              ║
║  42 Language Pairs: tr,en,de,fr,it,pt,es                  ║
║  Helsinki-NLP Models → Optimized ONNX (INT8)              ║
╚════════════════════════════════════════════════════════════╝
""")

def print_progress_bar(current, total, bar_length=40):
    """Print a progress bar"""
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r[{bar}] {current}/{total} ({percent*100:.1f}%)", end='', flush=True)

def download_model(source_lang, target_lang, model_options):
    """
    Step 1: Download PyTorch model from Hugging Face
    Returns: (success, model_name_used)
    """
    print(f"\n  [1/4] Downloading model...")
    
    # Create temp directory
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try each model option
    for idx, model_name in enumerate(model_options, 1):
        try:
            print(f"    Trying [{idx}/{len(model_options)}]: {model_name}")
            
            # Download model and tokenizer
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            
            # Save to temp directory
            model.save_pretrained(TEMP_DIR)
            tokenizer.save_pretrained(TEMP_DIR)
            
            print(f"    ✓ Downloaded: {model_name}")
            return True, model_name
            
        except Exception as e:
            error_msg = str(e)[:80]
            print(f"    ✗ Failed: {error_msg}...")
            if idx < len(model_options):
                print(f"    → Trying next option...")
            continue
    
    # All options failed
    print(f"    ✗ All model options failed for {source_lang}-{target_lang}")
    return False, None

def convert_to_onnx(output_dir):
    """
    Step 2: Convert PyTorch model to ONNX format
    Only keeps files required by onnx_translation package
    """
    print(f"  [2/4] Converting to ONNX...")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and convert to ONNX
        model = ORTModelForSeq2SeqLM.from_pretrained(
            TEMP_DIR,
            export=True,
            provider="CPUExecutionProvider"
        )
        
        # Load tokenizer
        try:
            tokenizer = MarianTokenizer.from_pretrained(TEMP_DIR)
        except:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(TEMP_DIR)
        
        # Save ONNX model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Clean up unnecessary files to save space
        # Keep only files required by onnx_translation package
        required_files = {
            'encoder_model.onnx',
            'decoder_model.onnx',
            'vocab.json',
            'tokenizer_config.json',
            'generation_config.json',
            'special_tokens_map.json',  # Optional but useful
            'metadata.json',  # Created by script
        }
        
        # Remove unnecessary files
        removed_count = 0
        for file in output_dir.glob('*'):
            if file.is_file() and file.name not in required_files:
                file.unlink()
                removed_count += 1
        
        if removed_count > 0:
            print(f"    → Removed {removed_count} unnecessary file(s) (space saved)")
        
        print(f"    ✓ Converted to ONNX (optimized)")
        return True
        
    except Exception as e:
        print(f"    ✗ Conversion failed: {str(e)[:80]}")
        return False

def quantize_model(output_dir):
    """
    Step 3: Apply aggressive INT8 quantization for mobile
    Only quantizes encoder and decoder (required files)
    """
    print(f"  [3/4] Applying INT8 quantization...")
    
    try:
        # Quantization configuration
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        
        # Only quantize required model files
        model_files = [
            ("encoder_model.onnx", "encoder"),
            ("decoder_model.onnx", "decoder"),
        ]
        
        quantized_dir = output_dir / "quantized"
        quantized_dir.mkdir(exist_ok=True)
        
        # Quantize each component
        for filename, name in model_files:
            filepath = output_dir / filename
            if not filepath.exists():
                continue
            
            try:
                quantizer = ORTQuantizer.from_pretrained(output_dir, file_name=filename)
                quantizer.quantize(
                    save_dir=quantized_dir,
                    quantization_config=qconfig,
                    file_suffix=f"_{name}"
                )
            except Exception as e:
                print(f"    ⚠ Could not quantize {name}: {str(e)[:50]}")
        
        # Replace with quantized versions
        for file in quantized_dir.glob("*.onnx"):
            # Clean up filename
            new_name = file.name.split("__")[0] + ".onnx" if "__" in file.name else file.name
            new_name = new_name.replace(f"_{name}", "") if f"_{name}" in new_name else new_name
            
            target = output_dir / new_name
            if target.exists():
                target.unlink()
            shutil.move(str(file), str(target))
        
        # Cleanup
        shutil.rmtree(quantized_dir, ignore_errors=True)
        
        # Remove duplicate files
        for old_file in output_dir.glob("*__*.onnx"):
            old_file.unlink()
        
        print(f"    ✓ Quantized (INT8) - Space optimized")
        return True
        
    except Exception as e:
        print(f"    ⚠ Quantization failed (will use unquantized): {str(e)[:50]}")
        return True  # Non-critical

def cleanup_temp():
    """
    Step 4: Delete PyTorch model to save disk space
    """
    print(f"  [4/4] Cleaning up temporary files...")
    
    try:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            print(f"    ✓ Deleted PyTorch model (disk space saved)")
        return True
    except Exception as e:
        print(f"    ⚠ Cleanup warning: {str(e)[:50]}")
        return True  # Non-critical

def get_model_size(directory):
    """Calculate total size of ONNX models in MB"""
    total_size = 0
    for file in Path(directory).rglob("*.onnx"):
        total_size += file.stat().st_size
    return total_size / (1024 * 1024)

def process_language_pair(source_lang, target_lang, index, total):
    """Process a single language pair"""
    info = get_language_pair_info(source_lang, target_lang)
    pair_name = f"{source_lang}-{target_lang}"
    
    # Check if already completed
    if pair_name in progress_data['completed']:
        print(f"\n[{index}/{total}] {info['source_name']} → {info['target_name']} - SKIPPED (already done)")
        progress_data['skipped'].append(pair_name)
        return True
    
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] Processing: {info['source_name']} → {info['target_name']} ({pair_name})")
    print(f"{'='*60}")
    
    progress_data['current'] = pair_name
    
    output_dir = OUTPUT_BASE_DIR / pair_name
    
    # Pipeline
    success, model_name = download_model(source_lang, target_lang, info['models'])
    if not success:
        progress_data['failed'].append({
            'pair': pair_name,
            'reason': 'download_failed'
        })
        save_progress()
        return False
    
    if not convert_to_onnx(output_dir):
        progress_data['failed'].append({
            'pair': pair_name,
            'reason': 'conversion_failed'
        })
        save_progress()
        cleanup_temp()
        return False
    
    quantize_model(output_dir)  # Non-critical
    cleanup_temp()  # Clean up immediately after conversion
    
    # Save metadata
    metadata = {
        'source_lang': source_lang,
        'target_lang': target_lang,
        'model_name': model_name,
        'converted_at': datetime.now().isoformat(),
        'size_mb': get_model_size(output_dir)
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Mark as completed
    progress_data['completed'].append(pair_name)
    save_progress()
    
    size = metadata['size_mb']
    print(f"\n✓ Completed: {pair_name} ({size:.1f} MB)")
    
    return True

def print_summary():
    """Print final summary"""
    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    completed = len(progress_data['completed'])
    failed = len(progress_data['failed'])
    skipped = len(progress_data['skipped'])
    total = progress_data['total_pairs']
    
    print(f"\n✓ Completed: {completed}/{total}")
    print(f"✗ Failed:    {failed}/{total}")
    print(f"⊘ Skipped:   {skipped}/{total}")
    
    if progress_data['failed']:
        print(f"\nFailed language pairs:")
        for item in progress_data['failed']:
            print(f"  - {item['pair']}: {item['reason']}")
    
    # Calculate total size
    total_size = 0
    for pair_dir in OUTPUT_BASE_DIR.glob("*-*"):
        if pair_dir.is_dir():
            total_size += get_model_size(pair_dir)
    
    print(f"\nTotal size: {total_size:.1f} MB (~{total_size/1024:.2f} GB)")
    print(f"Average per model: {total_size/completed:.1f} MB" if completed > 0 else "")
    
    print(f"\nOutput directory: {OUTPUT_BASE_DIR}")
    print(f"{'='*60}\n")

def main():
    """Main execution"""
    print_header()
    
    # Load previous progress
    load_progress()
    
    if progress_data['started_at'] is None:
        progress_data['started_at'] = datetime.now().isoformat()
    
    print(f"Processing {len(ALL_LANGUAGE_PAIRS)} language pairs...")
    print(f"Output: {OUTPUT_BASE_DIR}")
    print(f"Note: PyTorch models will be deleted after each conversion\n")
    
    # Process each language pair
    total = len(ALL_LANGUAGE_PAIRS)
    for index, (source, target) in enumerate(ALL_LANGUAGE_PAIRS, 1):
        process_language_pair(source, target, index, total)
        print_progress_bar(index, total)
    
    print("\n")  # New line after progress bar
    print_summary()
    
    print("✓ All processing complete!")
    print("Next steps:")
    print("  1. Run optimize_all_models.py for additional optimizations")
    print("  2. Run test_all_models.py to verify models")
    print("  3. Upload to Hugging Face for CDN access\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        print("Progress has been saved. Run again to resume from where you left off.")
        save_progress()
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        save_progress()
        sys.exit(1)
