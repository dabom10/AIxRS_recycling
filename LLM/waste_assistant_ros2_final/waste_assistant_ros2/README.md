# waste_assistant_ros2

ROS2 package for image + audio + bbox based waste reasoning with Gemini.

## Query input
- image_path
- audio_path
- detections: list of bbox candidates from YOLO

## Final decision output
- image_path
- classification_possible
- object_name
- selected_bbox_index
- selected_bbox_xyxy
- selected_center_xy
- all_bboxes
- needs_human_action
- human_action_reason
- robot_action
- bin_type
- answer

## Notes
- question 텍스트는 따로 사용하지 않음
- audio_path의 음성이 사용자 질문 역할을 함
- selected_center_xy는 원본 이미지 기준 픽셀 좌표임
