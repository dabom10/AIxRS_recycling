import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / '.env')
load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv('GEMINI_API_KEY', '')
    gemini_model: str = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    gemini_temperature: float = float(os.getenv('GEMINI_TEMPERATURE', '0.2'))
    move_threshold: float = float(os.getenv('MOVE_THRESHOLD', '0.85'))
    ask_threshold: float = float(os.getenv('ASK_THRESHOLD', '0.70'))


settings = Settings()
