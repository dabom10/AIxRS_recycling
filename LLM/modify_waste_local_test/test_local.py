import argparse
import json
from app.gemini_service import GeminiWasteAssistant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="이미지 파일 경로")
    parser.add_argument("--audio", default=None, help="음성 파일 경로")
    parser.add_argument("--vision-hint", default=None, help="YOLO 등 비전 힌트")
    args = parser.parse_args()

    assistant = GeminiWasteAssistant()
    result = assistant.ask(
        image_path=args.image,
        audio_path=args.audio,
        vision_hint=args.vision_hint,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
