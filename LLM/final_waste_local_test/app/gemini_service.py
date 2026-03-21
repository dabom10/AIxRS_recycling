import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    @staticmethod
    def _pick_default_detection(
        detections: List[Dict[str, Any]]
    ) -> tuple[Optional[int], Optional[List[int]], Optional[List[float]]]:
        if not detections:
            return None, None, None

        if len(detections) == 1:
            det = detections[0]
            return det.get("idx"), det.get("bbox_xyxy"), det.get("center")

        sorted_dets = sorted(detections, key=lambda d: float(d.get("dist2", 1e18)))
        det = sorted_dets[0]
        return det.get("idx"), det.get("bbox_xyxy"), det.get("center")

    def _mock_response(
        self,
        image_path: str,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        all_bboxes = detections or []
        selected_idx, selected_bbox, selected_center = self._pick_default_detection(all_bboxes)

        return {
            "image_path": image_path,
            "classification_possible": True,
            "object_name": "플라스틱병",
            "selected_bbox_index": selected_idx,
            "selected_bbox_xyxy": selected_bbox,
            "selected_center_xy": selected_center,
            "all_bboxes": all_bboxes,
            "needs_human_action": True,
            "human_action_reason": "라벨 제거 또는 세척 필요",
            "robot_action": "STOP",
            "bin_type": None,
            "answer": "사진상 플라스틱병으로 보입니다. 라벨 제거 또는 세척이 필요할 수 있어 지금은 로봇이 옮기지 않습니다.",
        }

    @staticmethod
    def _normalize_response(
        data: Dict[str, Any],
        image_path: str,
        detections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        data = dict(data)

        data.setdefault("image_path", image_path)
        data.setdefault("classification_possible", False)
        data.setdefault("object_name", None)
        data.setdefault("selected_bbox_index", None)
        data.setdefault("selected_bbox_xyxy", None)
        data.setdefault("selected_center_xy", None)
        data.setdefault("all_bboxes", detections)
        data.setdefault("needs_human_action", True)
        data.setdefault("human_action_reason", "판단 결과가 충분하지 않습니다.")
        data.setdefault("robot_action", "STOP")
        data.setdefault("bin_type", None)
        data.setdefault("answer", "이미지와 음성만으로는 확실한 판단이 어렵습니다.")

        det_map = {d.get("idx"): d for d in detections}
        idx = data.get("selected_bbox_index")

        if idx in det_map:
            det = det_map[idx]
            data["selected_bbox_xyxy"] = det.get("bbox_xyxy")
            data["selected_center_xy"] = det.get("center")
        elif len(detections) == 1:
            det = detections[0]
            data["selected_bbox_index"] = det.get("idx")
            data["selected_bbox_xyxy"] = det.get("bbox_xyxy")
            data["selected_center_xy"] = det.get("center")

        data["all_bboxes"] = detections
        return data

    def ask(
        self,
        image_path: str,
        audio_path: Optional[str] = None,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not image_path:
            raise ValueError("image_path는 필수입니다.")

        all_bboxes = detections or []

        if self.use_mock:
            return self._mock_response(
                image_path=image_path,
                detections=all_bboxes,
            )

        if self.client is None:
            raise RuntimeError("Gemini client가 초기화되지 않았습니다.")

        contents: List[Any] = [self._file_part_from_path(image_path)]

        if audio_path:
            contents.append(self._file_part_from_path(audio_path))

        prompt_payload = {
            "image_path": image_path,
            "all_bboxes": all_bboxes,
        }

        contents.append(json.dumps(prompt_payload, ensure_ascii=False, indent=2))

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            ),
        )

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini 응답 텍스트가 비어 있습니다.")

        data = json.loads(text)
        return self._normalize_response(
            data=data,
            image_path=image_path,
            detections=all_bboxes,
        )