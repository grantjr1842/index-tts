from typing import Dict, List

# Supported languages
LANGUAGES: Dict[str, str] = {
    "中文": "zh_CN",
    "English": "en_US"
}

# Emotion control choices (keys for i18n)
EMO_CHOICES_KEYS: List[str] = [
    "与音色参考音频相同",
    "使用情感参考音频",
    "使用情感向量控制",
    "使用情感描述文本控制"
]

# Application settings
MODE: str = 'local'
MAX_LENGTH_TO_USE_SPEED: int = 70
