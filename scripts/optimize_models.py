"""
Advanced ONNX Model Optimizer
Additional optimization utilities for mobile deployment
"""

import os
from pathlib import Path
import onnx
from onnx import optimizer
import json

OUTPUT_DIR = Path(__file__).parent.parent / "onnx_models" / "en-tr"

def optimize_onnx_graph():
    """Apply graph optimizations to ONNX models"""
    print("\n" + "="*60)
    print("ADVANCED OPTIMIZATION: Graph Optimization")
    print("="*60 + "\n")
    
    models = [
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx"
    ]
    
    for model_name in models:
        model_path = OUTPUT_DIR / model_name
        if not model_path.exists():
            continue
            
        print(f"Optimizing {model_name}...")
        
        try:
            # Load model
            model = onnx.load(str(model_path))
            
            # Apply optimization passes
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'fuse_bn_into_conv',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_transpose_into_gemm',
            ]
            
            optimized_model = optimizer.optimize(model, passes)
            
            # Save optimized model
            onnx.save(optimized_model, str(model_path))
            print(f"  ✓ {model_name} optimized")
            
        except Exception as e:
            print(f"  ⚠ Could not optimize {model_name}: {e}")
    
    print("\n✓ Graph optimization completed\n")

def create_model_metadata():
    """Create metadata file for the model"""
    print("Creating model metadata...")
    
    metadata = {
        "model_name": "Helsinki-NLP/opus-mt-en-tr",
        "source_language": "en",
        "target_language": "tr",
        "framework": "ONNX",
        "quantization": "INT8",
        "optimizations": [
            "Graph optimization",
            "Quantization (INT8)",
            "Mobile-optimized"
        ],
        "usage": {
            "package": "onnx_translation: ^0.1.2",
            "example": "See README.md for Flutter integration"
        },
        "files": {
            "encoder": "encoder_model.onnx",
            "decoder": "decoder_model.onnx",
            "decoder_with_past": "decoder_with_past_model.onnx",
            "tokenizer": "tokenizer.json",
            "config": "config.json"
        }
    }
    
    metadata_path = OUTPUT_DIR / "model_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadata saved to {metadata_path}\n")

def print_final_stats():
    """Print final model statistics"""
    print("\n" + "="*60)
    print("FINAL MODEL STATISTICS")
    print("="*60 + "\n")
    
    files = {
        "encoder_model.onnx": 0,
        "decoder_model.onnx": 0,
        "decoder_with_past_model.onnx": 0,
        "tokenizer.json": 0,
        "config.json": 0,
    }
    
    total_size = 0
    for filename in files.keys():
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            files[filename] = size_mb
            total_size += size_mb
            print(f"  {filename:30s} {size_mb:>8.2f} MB")
    
    print(f"\n  {'TOTAL SIZE':30s} {total_size:>8.2f} MB")
    print(f"\n  Original size (estimate):     ~300.00 MB")
    print(f"  Size reduction:               ~{((300 - total_size) / 300 * 100):.1f}%")
    print("\n" + "="*60 + "\n")

def main():
    """Main optimization flow"""
    print("""
╔════════════════════════════════════════════════════════════╗
║  ONNX Model Advanced Optimizer                            ║
║  Additional optimizations for mobile deployment           ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    if not OUTPUT_DIR.exists():
        print("✗ Error: Model directory not found. Run download_onnx_models.py first.")
        return
    
    # Run optimizations
    optimize_onnx_graph()
    create_model_metadata()
    print_final_stats()
    
    print("✓ Advanced optimization completed!\n")

if __name__ == "__main__":
    main()
