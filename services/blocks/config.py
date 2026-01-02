"""
Centralized configuration file
"""

class AppConfig:
    # AI Models
    GEMINI_MODELS = {
        "pro": "gemini-2.5-pro",
        "flash": "gemini-2.5-flash",
        "lite": "gemini-2.5-flash-lite"
    }
    
    # Translation
    SUPPORTED_LANGUAGES = ["Chinese", "English", "Vietnamese", "French", "Japanese", "Korean"]
    
    # Voice
    TTS_VOICES = {
        "vi": {"female": "vi-VN-HoaiMyNeural", "male": "vi-VN-NamMinhNeural"},
        "en": {"female": "en-US-EmmaNeural", "male": "en-US-AndrewMultilingualNeural"},
        "zh": {"female": "zh-CN-XiaoyiNeural", "male": "zh-CN-YunjianNeural"}
    }
    
    # Rate Limits
    API_LIMITS = {
        "gemini_daily": 1500,
        "translation_per_hour": 100,
        "tts_per_day": 500
    }
    
    # Cache
    CACHE_TTL = {
        "translation": 86400,  # 24h
        "book_analysis": 3600,  # 1h
        "embedding": 604800    # 7 days
    }
