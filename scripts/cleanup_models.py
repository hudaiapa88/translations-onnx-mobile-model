"""
Final cleanup script - Removes unnecessary files from ALL model directories
Keeps only files required by onnx_translation package for space optimization
Run this after download_all_languages.py to clean up all language pairs
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "onnx_models"

# Files required by onnx_translation package
REQUIRED_FILES = {
    'encoder_model.onnx',
    'decoder_model.onnx',
    'vocab.json',
    'tokenizer_config.json',
    'generation_config.json',
    'special_tokens_map.json',  # Optional but useful
    'metadata.json',  # Created by script
    'README.md',  # Documentation (if exists)
}

print("\n" + "="*60)
print("Final Cleanup - Space Optimization")
print("Removing unnecessary files from ALL model directories")
print("="*60 + "\n")

total_removed = 0
total_size_saved = 0

# Process all language pair directories
pair_dirs = sorted([d for d in MODELS_DIR.glob("*-*") if d.is_dir()])

if not pair_dirs:
    print("✗ No language pair directories found!")
    print(f"  Expected location: {MODELS_DIR}")
else:
    print(f"Found {len(pair_dirs)} language pair directories\n")
    
    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        removed_count = 0
        size_saved = 0
        
        # Check each file
        for file in pair_dir.glob("*"):
            if file.is_file() and file.name not in REQUIRED_FILES:
                size_mb = file.stat().st_size / (1024 * 1024)
                file.unlink()
                removed_count += 1
                size_saved += size_mb
        
        if removed_count > 0:
            print(f"✓ {pair_name}: Removed {removed_count} file(s), saved {size_saved:.1f} MB")
            total_removed += removed_count
            total_size_saved += size_saved

if total_removed > 0:
    print(f"\n{'='*60}")
    print(f"Total files removed: {total_removed}")
    print(f"Total space saved: {total_size_saved:.1f} MB ({total_size_saved/1024:.2f} GB)")
    print(f"{'='*60}\n")
else:
    print("\n✓ All directories are already optimized!\n")

print("✓ Cleanup complete! Only required files remain.\n")

# Show final statistics
print("Final Statistics:")
print("="*60)
total_size = 0
for pair_dir in pair_dirs:
    pair_size = sum(f.stat().st_size for f in pair_dir.glob("*.onnx"))
    total_size += pair_size

if pair_dirs:
    print(f"Total ONNX models size: {total_size / (1024**3):.2f} GB")
    print(f"Average per language pair: {(total_size / len(pair_dirs)) / (1024**2):.1f} MB")
print("="*60 + "\n")
