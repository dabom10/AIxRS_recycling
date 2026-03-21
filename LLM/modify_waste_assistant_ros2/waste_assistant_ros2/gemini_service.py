import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai
from google.genai import types

from .config import settings
from .models import WasteDecision
from .prompts import SYSTEM_INSTRUCTION


class GeminiWasteAssistant:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise ValueError('GEMINI_API_KEY가 설정되지 않았습니다.')
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model
        self.temperature = settings.gemini_temperature

    @staticmethod
    def _guess_mime_type(file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type
        ext = Path(file_path).suffix.lower()
        fallback = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp',
            '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4', '.aac': 'audio/aac',
            '.ogg': 'audio/ogg', '.flac': 'audio/flac',
        }
        if ext in fallback:
            return fallback[ext]
        raise ValueError(f'지원하지 않는 파일 형식입니다: {file_path}')

    def _part_from_path(self, file_path: str) -> types.Part:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f'파일이 존재하지 않습니다: {file_path}')
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=self._guess_mime_type(file_path))

    def ask(self, image_path: str, audio_path: Optional[str], question: Optional[str], vision_hint: Optional[Dict[str, Any]]) -> WasteDecision:
        contents = [self._part_from_path(image_path)]
        if audio_path:
            contents.append(self._part_from_path(audio_path))

        prompt_lines = []
        if question:
            prompt_lines.append(f'사용자 질문: {question}')
        if vision_hint:
            prompt_lines.append(f'추가 비전 정보: {vision_hint}')
        prompt_lines.append('반드시 지정된 JSON 스키마에 맞춰 답변하라.')
        contents.append('\n'.join(prompt_lines))

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type='application/json',
                response_schema=WasteDecision,
            ),
        )
        return WasteDecision.model_validate_json(response.text)
