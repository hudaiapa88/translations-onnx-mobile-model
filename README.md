# Multi-Language ONNX Translation Models

Optimized ONNX translation models for mobile deployment. Supports 42 language pairs across 7 languages: Turkish, English, German, French, Italian, Portuguese, and Spanish.

## 📋 Supported Languages

- 🇹🇷 Turkish (tr)
- 🇬🇧 English (en)
- 🇩🇪 German (de)
- 🇫🇷 French (fr)
- 🇮🇹 Italian (it)
- 🇵🇹 Portuguese (pt)
- 🇪🇸 Spanish (es)

**Total Language Pairs:** 42 (each language can translate to all others)

## 🚀 Quick Start

### 1. Setup and Download Models

```bash
cd scripts
run_all.bat
```

This will:
- Install Python dependencies
- Download all 42 language pairs from Helsinki-NLP
- Convert to ONNX format with INT8 quantization
- Optimize for mobile deployment
- Test all models
- Delete temporary PyTorch files automatically

**Estimated Time:** 4-6 hours  
**Disk Space Required:** ~5-7 GB (final optimized models)

### 2. Manual Steps (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Download and convert all models
python download_all_languages.py

# Optimize models
python optimize_all_models.py

# Test all models
python test_all_models.py
```

## 📁 Project Structure

```
translations-onnx-mobile-model/
├── onnx_models/              # Output directory
│   ├── en-tr/                # English → Turkish
│   │   ├── encoder_model.onnx          ✅ Required
│   │   ├── decoder_model.onnx          ✅ Required
│   │   ├── vocab.json                  ✅ Required
│   │   ├── tokenizer_config.json       ✅ Required
│   │   ├── generation_config.json      ✅ Required
│   │   ├── special_tokens_map.json     (Optional)
│   │   └── metadata.json               (Created by script)
│   ├── tr-en/                # Turkish → English
│   ├── en-de/                # English → German
│   └── ...                   # (42 total pairs)
├── scripts/
│   ├── language_config.py           # Language pair configuration
│   ├── download_all_languages.py    # Main download/convert script
│   ├── optimize_all_models.py       # Additional optimizations
│   ├── test_all_models.py           # Test all models
│   ├── run_all.bat                  # Automated setup (Windows)
│   └── requirements.txt             # Python dependencies
├── download_log.json         # Progress tracking
├── test_results.json         # Test results
└── README.md
```

## 📊 Model Specifications

- **Format:** ONNX (Open Neural Network Exchange)
- **Quantization:** INT8 (75% size reduction)
- **Framework:** Helsinki-NLP Opus-MT models
- **Files per model:** 5-6 files (only required by onnx_translation package)
- **Optimizations:**
  - Dynamic quantization
  - Graph optimization
  - Operator fusion
  - Unnecessary file removal
- **Average Model Size:** ~60-80 MB per language pair (optimized)
- **Total Size:** ~2.5-3.5 GB (all 42 pairs, space optimized)

## 🌐 Deployment Options

### Option 1: Hugging Face (Recommended)

Upload models to Hugging Face for free CDN access:

```bash
# Login to Hugging Face
huggingface-cli login

# Upload all models
huggingface-cli upload [username]/translation-models ./onnx_models .
```

Access via URL:
```
https://huggingface.co/[username]/translation-models/resolve/main/en-tr/encoder_model.onnx
```

### Option 2: GitHub Releases

1. Create a new release on GitHub
2. Compress language pairs: `zip -r en-tr.zip onnx_models/en-tr/`
3. Upload ZIP files as release assets

Access via URL:
```
https://github.com/[username]/[repo]/releases/download/v1.0.0/en-tr.zip
```

### Option 3: GitHub Raw (Limited)

For smaller models (<100MB), use GitHub raw content:
```
https://raw.githubusercontent.com/[username]/[repo]/main/onnx_models/en-tr/config.json
```

## 📱 Flutter Integration

### Add Dependencies

```yaml
dependencies:
  onnx_translation: ^0.1.2
  http: ^1.1.0
  path_provider: ^2.1.1
```

### Download and Use Models

```dart
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'dart:io';

Future<void> downloadModel(String sourceLang, String targetLang) async {
  final baseUrl = 'https://huggingface.co/[username]/translation-models/resolve/main';
  final langPair = '$sourceLang-$targetLang';
  
  final files = [
    'encoder_model.onnx',           // Required
    'decoder_model.onnx',           // Required
    'vocab.json',                   // Required
    'tokenizer_config.json',        // Required
    'generation_config.json',       // Required
    'special_tokens_map.json',      // Optional
  ];
  
  final appDir = await getApplicationDocumentsDirectory();
  final modelDir = Directory('${appDir.path}/models/$langPair');
  await modelDir.create(recursive: true);
  
  for (var fileName in files) {
    final url = '$baseUrl/$langPair/$fileName';
    final response = await http.get(Uri.parse(url));
    
    if (response.statusCode == 200) {
      final file = File('${modelDir.path}/$fileName');
      await file.writeAsBytes(response.bodyBytes);
      print('Downloaded: $fileName');
    }
  }
  
  print('Model ready: ${modelDir.path}');
}

// Usage
await downloadModel('en', 'tr');  // English → Turkish
await downloadModel('tr', 'de');  // Turkish → German
```

## 🧪 Testing

View test results:
```bash
cat test_results.json
```

Test individual language pair:
```bash
python -c "from test_all_models import test_translation; test_translation('onnx_models/en-tr', 'en', 'tr')"
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Model Size (per pair) | ~100 MB |
| Size Reduction | ~75% (vs Float32) |
| Inference Speed | 2-3x faster (vs Float32) |
| Accuracy Loss | <2% |
| Supported Platforms | iOS, Android, Web, Desktop |

## 🔧 Troubleshooting

### Download Failed
- Check internet connection
- Some language pairs may not have pre-trained models
- Review `download_log.json` for failed pairs
- Re-run script - it will resume from last successful download

### Out of Disk Space
- Script automatically deletes PyTorch models after conversion
- Each model temporarily needs ~500 MB during conversion
- Final size: ~100 MB per language pair

### Model Not Working
- Check `test_results.json` for errors
- Verify all required files exist in model directory
- Try re-downloading the specific language pair

### Optimization Errors
- Optimization is non-critical
- Models will work even if optimization fails
- Check onnxoptimizer installation

## 📝 Progress Tracking

The download script automatically saves progress in `download_log.json`. If interrupted:

```bash
# Resume from where it left off
python download_all_languages.py
```

Progress data includes:
- Completed language pairs
- Failed pairs with error messages
- Current pair being processed
- Start time and duration

## 🎯 Use Cases

- **Mobile Translation Apps:** Offline translation in Flutter/React Native apps
- **IoT Devices:** Translation on edge devices
- **Privacy-Focused Apps:** No data sent to external servers
- **Multilingual Chatbots:** Local translation capabilities
- **Language Learning Apps:** Practice translations offline

## 🤝 Contributing

To add more languages:
1. Edit `scripts/language_config.py`
2. Add language code to `LANGUAGES` list
3. Add language name to `LANGUAGE_NAMES` dict
4. Run `download_all_languages.py`

## 📄 License

Models are from Helsinki-NLP (Apache 2.0)  
Scripts: MIT License

## 🙏 Credits

- **Models:** [Helsinki-NLP](https://huggingface.co/Helsinki-NLP) Opus-MT
- **Framework:** [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- **Optimization:** [ONNX Runtime](https://onnxruntime.ai/)
- **Flutter Package:** [onnx_translation](https://pub.dev/packages/onnx_translation)

## 📞 Support

- Open an issue on GitHub
- Check `test_results.json` for diagnostics
- Review Helsinki-NLP model documentation

---

**Ready to start?** Run `scripts\run_all.bat` and grab a coffee ☕ (it'll take a few hours!)
