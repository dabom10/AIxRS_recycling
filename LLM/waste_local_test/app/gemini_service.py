import json
import mimetypes
from pathlib import Path
from typing import Optional, Any

from google import genai
from google.genai import types

from app.config import settings
from app.prompts import SYSTEM_INSTRUCTION


class GeminiWasteAssistant:
    def __init__(self) -> None:
        self.model = settings.gemini_model
        self.temperature = settings.gemini_temperature
        self.use_mock = settings.use_mock
        self.client = None if self.use_mock else genai.Client(api_key=settings.gemini_api_key)

    @staticmethod
    def _guess_mime_type(file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type
        ext = Path(file_path).suffix.lower()
        fallback_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        if ext in fallback_map:
            return fallback_map[ext]
        raise ValueError(f"MIME 타입을 추론할 수 없습니다: {file_path}")

    def _file_part_from_path(self, file_path: str) -> types.Part:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")
        mime_type = self._guess_mime_type(file_path)
        data = path.read_bytes()
        return types.Part.from_bytes(data=data, mime_type=mime_type)

    def _mock_response(self, text_question: Optional[str]) -> dict[str, Any]:
        q = (text_question or "").strip()
        return {
            "object_name": "플라스틱병",
            "needs_human_action": True,
            "human_action_reason": "라벨 제거 또는 세척 필요",
            "robot_action": "STOP",
            "bin_type": None,
            "answer": f"사진상 플라스틱병으로 보입니다. {q or '추가 작업이 필요해'} 라벨 제거 또는 세척이 필요할 수 있어 지금은 로봇이 옮기지 않습니다."
        }

    def ask(
        self,
        image_path: str,
        audio_path: Optional[str] = None,
        text_question: Optional[str] = None,
        vision_hint: Optional[str] = None,
    ) -> dict[str, Any]:
        if not image_path:
            raise ValueError("image_path는 필수입니다.")
        if not audio_path and not text_question:
            raise ValueError("audio_path 또는 text_question 중 하나는 반드시 필요합니다.")

        if self.use_mock:
            return self._mock_response(text_question)

        contents = [self._file_part_from_path(image_path)]
        if audio_path:
            contents.append(self._file_part_from_path(audio_path))

        prompt_parts = []
        if text_question:
            prompt_parts.append(f"사용자 질문: {text_question}")
        if vision_hint:
            prompt_parts.append(f"추가 비전 분석 정보: {vision_hint}")
        prompt_parts.append("이미지와 음성을 함께 참고해서 결과 JSON만 출력해줘.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents + ["\n".join(prompt_parts)],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            ),
        )

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini 응답 텍스트가 비어 있습니다.")
        return json.loads(text)
