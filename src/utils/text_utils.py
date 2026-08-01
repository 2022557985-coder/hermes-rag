"""Text processing utilities: encoding detection, tokenization routing."""

import re
from typing import List, Optional

import charset_normalizer


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using charset_normalizer.

    Args:
        file_path: Path to the file.

    Returns:
        Detected encoding string (e.g., 'utf-8', 'gb2312').
    """
    with open(file_path, "rb") as f:
        raw = f.read(10000)
    result = charset_normalizer.detect(raw)
    return result.get("encoding") or "utf-8"


def read_file_with_encoding(file_path: str) -> str:
    """Read a file with automatic encoding detection.

    Args:
        file_path: Path to the file.

    Returns:
        File content as string.
    """
    encoding = detect_encoding(file_path)
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


def is_chinese_char(char: str) -> bool:
    """Check if a character is CJK (Chinese, Japanese, Korean)."""
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)  # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)  # CJK Unified Ideographs Extension A
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Unified Ideographs Extension B
        or (0xF900 <= cp <= 0xFAFF)  # CJK Compatibility Ideographs
    )


def is_english_char(char: str) -> bool:
    """Check if a character is ASCII letter."""
    return char.isascii() and char.isalpha()


def get_language_ratio(text: str) -> dict:
    """Calculate the ratio of Chinese vs English characters in text.

    Returns:
        dict with 'chinese_ratio' and 'english_ratio'.
    """
    if not text:
        return {"chinese_ratio": 0.0, "english_ratio": 0.0}

    total = len(text)
    chinese_count = sum(1 for c in text if is_chinese_char(c))
    english_count = sum(1 for c in text if is_english_char(c))

    return {
        "chinese_ratio": chinese_count / total if total > 0 else 0.0,
        "english_ratio": english_count / total if total > 0 else 0.0,
    }


def tokenize_text(text: str) -> List[str]:
    """Route text to the appropriate tokenizer based on language.

    Chinese text -> jieba
    English text -> nltk.word_tokenize
    Mixed text -> combined

    Args:
        text: Input text.

    Returns:
        List of tokens.
    """
    lang = get_language_ratio(text)

    if lang["chinese_ratio"] > 0.3:
        # Chinese-dominant: use jieba
        try:
            import jieba

            return list(jieba.cut(text))
        except ImportError:
            return text.split()

    if lang["english_ratio"] > 0.7:
        # English-dominant: use nltk
        try:
            import nltk

            try:
                return nltk.word_tokenize(text)
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                return nltk.word_tokenize(text)
        except ImportError:
            return text.split()

    # Mixed: use jieba for Chinese parts, nltk for English
    tokens = []
    try:
        import jieba

        tokens = list(jieba.cut(text))
    except ImportError:
        tokens = text.split()

    return [t for t in tokens if t.strip()]


def clean_text(text: str) -> str:
    """Clean text by removing excessive whitespace and control characters.

    Args:
        text: Input text.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    # Remove control characters (except newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def estimate_tokens(text: str) -> int:
    """Roughly estimate the number of tokens in text.

    Uses a simple heuristic: ~4 characters per token for English,
    ~1.5 characters per token for Chinese.

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    lang = get_language_ratio(text)
    chinese_chars = int(len(text) * lang["chinese_ratio"])
    other_chars = len(text) - chinese_chars

    return int(chinese_chars / 1.5 + other_chars / 4.0)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling both Chinese and English punctuation.

    Args:
        text: Input text.

    Returns:
        List of sentences.
    """
    # Split on sentence-ending punctuation for both languages
    pattern = r"(?<=[。！？.!?\n])\s*"
    sentences = re.split(pattern, text)

    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]