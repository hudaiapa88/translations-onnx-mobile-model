"""
Language Configuration for Multi-Language Translation Models
Defines all language pairs and their Helsinki-NLP model names
"""

# Supported languages
LANGUAGES = ['tr', 'en', 'de', 'fr', 'it', 'pt', 'es']

# Language names (for display)
LANGUAGE_NAMES = {
    'tr': 'Turkish',
    'en': 'English',
    'de': 'German',
    'fr': 'French',
    'it': 'Italian',
    'pt': 'Portuguese',
    'es': 'Spanish',
}

def get_model_name(source_lang, target_lang):
    """
    Get Helsinki-NLP model name for a language pair
    Returns multiple options in priority order (some models may not exist)
    """
    # Try different model naming patterns
    model_options = [
        f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}",
        f"Helsinki-NLP/opus-mt-tc-big-{source_lang}-{target_lang}",
        f"Helsinki-NLP/opus-tatoeba-{source_lang}-{target_lang}",
    ]
    
    # Special cases for multi-language models
    if source_lang == 'en':
        # English to Romance languages often grouped
        if target_lang in ['es', 'fr', 'it', 'pt']:
            model_options.insert(0, f"Helsinki-NLP/opus-mt-en-ROMANCE")
    
    # English source often has "tc-big" variants
    if source_lang == 'en' or target_lang == 'en':
        model_options.insert(0, f"Helsinki-NLP/opus-mt-tc-big-{source_lang}-{target_lang}")
    
    return model_options

def get_all_language_pairs():
    """
    Generate all language pairs (source -> target)
    Each language can translate to all other languages
    """
    pairs = []
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source != target:
                pairs.append((source, target))
    return pairs

def get_language_pair_info(source_lang, target_lang):
    """Get display information for a language pair"""
    return {
        'source': source_lang,
        'target': target_lang,
        'source_name': LANGUAGE_NAMES.get(source_lang, source_lang),
        'target_name': LANGUAGE_NAMES.get(target_lang, target_lang),
        'dir_name': f"{source_lang}-{target_lang}",
        'models': get_model_name(source_lang, target_lang),
    }

# Generate all 42 language pairs
ALL_LANGUAGE_PAIRS = get_all_language_pairs()

if __name__ == "__main__":
    print(f"\nTotal language pairs: {len(ALL_LANGUAGE_PAIRS)}\n")
    print("Language pairs to be downloaded:")
    print("=" * 60)
    
    for i, (src, tgt) in enumerate(ALL_LANGUAGE_PAIRS, 1):
        info = get_language_pair_info(src, tgt)
        print(f"{i:2d}. {info['source_name']:12s} → {info['target_name']:12s} ({src}-{tgt})")
    
    print("=" * 60)
    print(f"\nTotal: {len(ALL_LANGUAGE_PAIRS)} models will be processed")
