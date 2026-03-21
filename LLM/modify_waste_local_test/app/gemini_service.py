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

    def _mock_response(self, vision_hint: Optional[str] = None) -> dict[str, Any]:
        hint_text = f" 추가 비전 정보: {vision_hint}" if vision_hint else ""
        return {
            "object_name": "플라스틱병",
            "needs_human_action": True,
            "human_action_reason": "라벨 제거 또는 세척 필요",
            "robot_action": "STOP",
            "bin_type": None,
            "answer": f"사진상 플라스틱병으로 보입니다.{hint_text} 라벨 제거 또는 세척이 필요할 수 있어 지금은 로봇이 옮기지 않습니다."
        }

    def ask(
        self,
        image_path: str,
        audio_path: str,
        vision_hint: Optional[str] = None,
    ) -> dict[str, Any]:
        if not image_path:
            raise ValueError("image_path는 필수입니다.")
        if not audio_path:
            raise ValueError("audio_path는 필수입니다.")

        if self.use_mock:
            return self._mock_response(vision_hint)

        contents = [
            self._file_part_from_path(image_path),
            self._file_part_from_path(audio_path),
        ]

        prompt_parts = []
        if vision_hint:
            prompt_parts.append(f"추가 비전 분석 정보: {vision_hint}")

        prompt_parts.append(
            "사용자의 음성 질문과 이미지를 함께 참고해서 분리배출 판단을 수행해줘."
        )
        prompt_parts.append(
            "음성에는 '이거 어디다 버려?', '이거 그대로 버려도 돼?' 같은 짧은 질문이 들어있다."
        )
        prompt_parts.append(
            "반드시 JSON만 출력해."
        )

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