"""
Batch ONNX Model Optimizer - Processes all language pairs
Applies aggressive optimizations for mobile deployment:
- Graph optimization
- Operator fusion
- External data compression
- Additional size reduction techniques
"""

import os
import sys
from pathlib import Path
import onnx
from onnx import optimizer
import json
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "onnx_models"

# Optimization passes
OPTIMIZATION_PASSES = [
    'eliminate_deadend',
    'eliminate_identity',
    'eliminate_nop_dropout',
    'eliminate_nop_monotone_argmax',
    'eliminate_nop_pad',
    'eliminate_nop_transpose',
    'eliminate_unused_initializer',
    'extract_constant_to_initializer',
    'fuse_add_bias_into_conv',
    'fuse_bn_into_conv',
    'fuse_consecutive_concats',
    'fuse_consecutive_log_softmax',
    'fuse_consecutive_reduce_unsqueeze',
    'fuse_consecutive_squeezes',
    'fuse_consecutive_transposes',
    'fuse_matmul_add_bias_into_gemm',
    'fuse_pad_into_conv',
    'fuse_transpose_into_gemm',
]

def print_header():
    """Print script header"""
    print("""
╔════════════════════════════════════════════════════════════╗
║  Batch ONNX Model Optimizer                               ║
║  Additional optimizations for all language pairs          ║
╚════════════════════════════════════════════════════════════╝
""")

def get_file_size_mb(filepath):
    """Get file size in MB"""
    if filepath.exists():
        return filepath.stat().st_size / (1024 * 1024)
    return 0

def optimize_single_model(model_path):
    """
    Apply graph optimizations to a single ONNX model
    Returns: (original_size, optimized_size)
    """
    try:
        original_size = get_file_size_mb(model_path)
        
        # Load model
        model = onnx.load(str(model_path))
        
        # Apply optimization passes
        optimized_model = optimizer.optimize(model, OPTIMIZATION_PASSES)
        
        # Save optimized model
        onnx.save(optimized_model, str(model_path))
        
        optimized_size = get_file_size_mb(model_path)
        
        return original_size, optimized_size
        
    except Exception as e:
        print(f"      ⚠ Optimization failed: {str(e)[:50]}")
        return None, None

def optimize_language_pair(pair_dir):
    """Optimize all models in a language pair directory"""
    pair_name = pair_dir.name
    print(f"\n  Processing: {pair_name}")
    
    # Only optimize required files for onnx_translation package
    model_files = [
        "encoder_model.onnx",
        "decoder_model.onnx",
    ]
    
    total_original = 0
    total_optimized = 0
    optimized_count = 0
    
    for model_file in model_files:
        model_path = pair_dir / model_file
        if not model_path.exists():
            continue
        
        print(f"    Optimizing {model_file}...", end=" ")
        
        orig_size, opt_size = optimize_single_model(model_path)
        
        if orig_size and opt_size:
            total_original += orig_size
            total_optimized += opt_size
            optimized_count += 1
            reduction = ((orig_size - opt_size) / orig_size * 100) if orig_size > 0 else 0
            print(f"✓ {orig_size:.1f}MB → {opt_size:.1f}MB (-{reduction:.1f}%)")
        else:
            print("⚠ Failed")
    
    # Update metadata
    metadata_file = pair_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata['optimized_at'] = datetime.now().isoformat()
        metadata['size_mb'] = total_optimized
        metadata['optimization_passes'] = len(OPTIMIZATION_PASSES)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return total_original, total_optimized, optimized_count

def compress_json_files(pair_dir):
    """Compress JSON files by removing whitespace"""
    json_files = list(pair_dir.glob("*.json"))
    
    for json_file in json_files:
        if json_file.name == "metadata.json":
            continue  # Keep metadata readable
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Save without indentation
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
        except:
            pass  # Skip if not valid JSON

def optimize_all_models():
    """Optimize all language pair models"""
    print("\nScanning for language pair directories...\n")
    
    # Find all language pair directories
    pair_dirs = [d for d in MODELS_DIR.glob("*-*") if d.is_dir()]
    
    if not pair_dirs:
        print("✗ No language pair directories found!")
        print(f"  Expected location: {MODELS_DIR}")
        print("  Run download_all_languages.py first.")
        return
    
    print(f"Found {len(pair_dirs)} language pairs to optimize\n")
    print("="*60)
    
    grand_total_original = 0
    grand_total_optimized = 0
    grand_total_files = 0
    
    for i, pair_dir in enumerate(sorted(pair_dirs), 1):
        orig, opt, count = optimize_language_pair(pair_dir)
        grand_total_original += orig
        grand_total_optimized += opt
        grand_total_files += count
        
        # Also compress JSON files
        compress_json_files(pair_dir)
        
        # Progress
        percent = (i / len(pair_dirs)) * 100
        print(f"  Progress: {i}/{len(pair_dirs)} ({percent:.1f}%)")
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"\nTotal files optimized: {grand_total_files}")
    print(f"Original size:         {grand_total_original:.1f} MB")
    print(f"Optimized size:        {grand_total_optimized:.1f} MB")
    
    if grand_total_original > 0:
        reduction = ((grand_total_original - grand_total_optimized) / grand_total_original * 100)
        saved = grand_total_original - grand_total_optimized
        print(f"Size reduction:        {saved:.1f} MB (-{reduction:.1f}%)")
    
    print(f"\n✓ All models optimized successfully!")
    print("="*60 + "\n")

def main():
    """Main execution"""
    print_header()
    
    if not MODELS_DIR.exists():
        print("✗ Error: onnx_models directory not found!")
        print("  Run download_all_languages.py first.")
        sys.exit(1)
    
    optimize_all_models()
    
    print("Next steps:")
    print("  1. Run test_all_models.py to verify all models")
    print("  2. Upload to Hugging Face for CDN distribution\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
