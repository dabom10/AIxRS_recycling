from typing import List, Literal, Optional
from pydantic import BaseModel, Field, conlist, confloat


class DetectionCandidate(BaseModel):
    idx: int = Field(ge=0)
    bbox_xyxy: conlist(float, min_length=4, max_length=4)
    center: conlist(float, min_length=2, max_length=2)
    class_name: Optional[str] = None
    conf: Optional[confloat(ge=0.0, le=1.0)] = None


class WasteAssistantDecision(BaseModel):
    image_path: str = Field(min_length=1)
    classification_possible: bool
    object_name: Optional[str] = None
    selected_bbox_index: Optional[int] = Field(default=None, ge=0)
    selected_bbox_xyxy: Optional[conlist(float, min_length=4, max_length=4)] = None
    selected_center_xy: Optional[conlist(float, min_length=2, max_length=2)] = None
    all_bboxes: List[DetectionCandidate] = Field(default_factory=list)
    needs_human_action: bool
    human_action_reason: Optional[str] = None
    robot_action: Literal['STOP', 'SORT']
    bin_type: Optional[Literal['can', 'paper', 'plastic', 'general']] = None
    answer: str = Field(min_length=1)
