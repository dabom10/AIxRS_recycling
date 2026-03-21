# grasp_detection

ROS 2 Python package for ROI-based grasp detection using YOLO segmentation.

## Topics
- Subscribe: `/camera/image_raw`
- Publish detections: `/grasp_detection/detections`
- Publish debug image: `/grasp_detection/debug_image`

## Notes
- Put your model file in `model/waste4.pt`.
- The node crops a configured ROI, runs YOLO on the ROI, and publishes bounding boxes in full-image pixel coordinates.
