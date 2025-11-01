"""
Cleanup script to fix quantized models and remove duplicates
"""

import shutil
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "onnx_models" / "en-tr"

print("\n" + "="*60)
print("Cleaning up ONNX model directory")
print("="*60 + "\n")

# Files to keep (quantized versions)
keep_files = [
    "encoder_model__encoder.onnx",
    "decoder_model__decoder.onnx", 
    "decoder_with_past_model__decoder_with_past.onnx",
]

# Target names
target_names = [
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
]

# Rename quantized files to standard names
for old_name, new_name in zip(keep_files, target_names):
    old_path = OUTPUT_DIR / old_name
    new_path = OUTPUT_DIR / new_name
    
    if old_path.exists():
        if new_path.exists():
            new_path.unlink()
            print(f"✓ Removed old: {new_name}")
        
        shutil.move(str(old_path), str(new_path))
        print(f"✓ Renamed: {old_name} → {new_name}")

# Remove any other .onnx files (duplicates)
for file in OUTPUT_DIR.glob("*.onnx"):
    if file.name not in target_names:
        size_mb = file.stat().st_size / (1024 * 1024)
        file.unlink()
        print(f"✓ Deleted duplicate: {file.name} ({size_mb:.1f} MB)")

# Calculate final size
total_size = 0
print("\nFinal model files:")
for file in sorted(OUTPUT_DIR.glob("*.onnx")):
    size_mb = file.stat().st_size / (1024 * 1024)
    total_size += size_mb
    print(f"  {file.name:40s} {size_mb:>8.2f} MB")

print(f"\n{'Total size:':40s} {total_size:>8.2f} MB")
print("\n✓ Cleanup complete!\n")
