import argparse
import json

from app.gemini_service import GeminiWasteAssistant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="이미지 파일 경로")
    parser.add_argument("--audio", default=None, help="음성 파일 경로")
    parser.add_argument("--detections-file", default=None, help="detections JSON 파일 경로")
    parser.add_argument("--detections-json", default=None, help="detections JSON 문자열")

    args = parser.parse_args()

    detections = None

    if args.detections_file:
        with open(args.detections_file, "r", encoding="utf-8") as f:
            detections = json.load(f)
    elif args.detections_json:
        detections = json.loads(args.detections_json)

    assistant = GeminiWasteAssistant()
    result = assistant.ask(
        image_path=args.image,
        audio_path=args.audio,
        detections=detections,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()