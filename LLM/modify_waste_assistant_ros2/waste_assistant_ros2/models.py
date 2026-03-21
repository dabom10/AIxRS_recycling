from typing import List, Literal, Optional
from pydantic import BaseModel, Field, confloat


class WasteDecision(BaseModel):
    action: Literal['move', 'no_move']
    target_bin: Literal['can', 'paper', 'plastic', 'general', 'unknown']
    confidence: confloat(ge=0.0, le=1.0)
    preconditions: List[Literal['none', 'empty', 'rinse', 'remove_label', 'separate_cap', 'flatten']]
    clarify_question: Optional[str] = None
    speak: str = Field(min_length=1)
    reason: str = Field(min_length=1)
