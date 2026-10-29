"""Report languages: Chinese source text, English by phrase translation."""
from functools import lru_cache
import json
from pathlib import Path
import re

TEMPLATES = Path(__file__).resolve().parent / "templates"


@lru_cache(maxsize=1)
def english_translations():
    translations = json.loads((TEMPLATES / "en.json").read_text(encoding="utf-8"))
    # Match longer phrases first, and substitute once so translations cannot
    # translate each other. Only presentation text is passed to this helper.
    pattern = re.compile("|".join(re.escape(key) for key in sorted(translations, key=len, reverse=True)))
    return translations, pattern


def localize(text, language="zh"):
    if language == "zh" or text is None:
        return text
    if language != "en":
        raise ValueError(f"Unsupported report language: {language}")
    translations, pattern = english_translations()
    return pattern.sub(lambda match: translations[match.group()], text)
