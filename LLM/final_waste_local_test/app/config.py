import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    use_mock: bool = _to_bool(os.getenv("USE_MOCK", "false"))

    def validate(self) -> None:
        if not self.use_mock and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. 로컬 테스트만 하려면 USE_MOCK=true로 두세요.")


settings = Settings()
settings.validate()
