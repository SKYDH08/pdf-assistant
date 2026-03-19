from transformers import MarianMTModel, MarianTokenizer

# Supported languages and their Helsinki-NLP model names
LANGUAGES = {
    "Hindi":   "Helsinki-NLP/opus-mt-en-hi",
    "French":  "Helsinki-NLP/opus-mt-en-fr",
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
}

# Cache loaded models so we don't reload every time
_model_cache = {}


def get_translator(language: str):
    """
    Loads the MarianMT model and tokenizer directly.
    Downloads on first use, cached after that.
    """
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Choose from {list(LANGUAGES.keys())}")

    if language not in _model_cache:
        model_name = LANGUAGES[language]
        print(f"Loading translation model for {language}...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _model_cache[language] = (tokenizer, model)
        print(f"{language} model loaded.")

    return _model_cache[language]


def translate(text: str, language: str) -> str:
    """
    Translates English text into the target language.
    """
    if language == "English":
        return text

    tokenizer, model = get_translator(language)

    # Split into sentences to avoid token limit
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    translated_sentences = []
    for sentence in sentences:
        inputs = tokenizer([sentence], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_sentences.append(result)

    return ". ".join(translated_sentences)


def translate_all(text: str, languages: list[str]) -> dict[str, str]:
    """
    Translates text into multiple languages at once.
    Returns a dict like {"Hindi": "...", "French": "..."}
    """
    results = {"English": text}

    for lang in languages:
        if lang == "English":
            continue
        print(f"Translating to {lang}...")
        results[lang] = translate(text, lang)

    return results


if __name__ == "__main__":
    test_summary = """King penguins are surprisingly adapting to climate change due to their flexibility.
    While warming disrupts most animal reproduction cycles, penguins are a rare exception.
    Scientists are studying this unique case to understand animal resilience."""

    print("Original (English):")
    print(test_summary)

    for lang in ["Hindi", "French", "Spanish"]:
        print(f"\nTranslating to {lang}...")
        result = translate(test_summary, lang)
        print(f"{lang}:\n{result}")