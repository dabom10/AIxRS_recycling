from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from .config import settings


PRECONDITION_BLOCKLIST: Set[str] = {'empty', 'rinse', 'remove_label'}
OPTIONAL_BLOCKLIST: Set[str] = {'separate_cap', 'flatten'}


@dataclass
class VisionHint:
    object: str = 'unknown'
    subtype: str = 'unknown'
    confidence: float = 0.0
    has_label: bool = False
    dirty: bool = False
    liquid_inside: bool = False

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'VisionHint':
        data = data or {}
        return cls(
            object=str(data.get('object', 'unknown')),
            subtype=str(data.get('subtype', 'unknown')),
            confidence=float(data.get('confidence', 0.0)),
            has_label=bool(data.get('has_label', False)),
            dirty=bool(data.get('dirty', False)),
            liquid_inside=bool(data.get('liquid_inside', False)),
        )


def map_vision_to_bin(vision: VisionHint) -> str:
    obj = vision.object.lower()
    if obj in {'can', 'aluminum_can', 'metal_can'}:
        return 'can'
    if obj in {'paper', 'cardboard', 'paper_pack'}:
        return 'paper'
    if obj in {'plastic_bottle', 'pet_bottle', 'plastic_container', 'plastic'}:
        return 'plastic'
    if obj in {'general', 'trash', 'waste'}:
        return 'general'
    return 'unknown'


def derive_preconditions_from_vision(vision: VisionHint) -> Set[str]:
    found: Set[str] = set()
    if vision.has_label:
        found.add('remove_label')
    if vision.dirty:
        found.add('rinse')
    if vision.liquid_inside:
        found.add('empty')
    if 'transparent_pet' in vision.subtype.lower():
        found.add('flatten')
    return found


def fuse_confidence(p_v: float, p_l: float, agree: bool) -> float:
    base = min(max(p_v, 0.0), max(p_l, 0.0))
    return base if agree else 0.5 * base


def gate_and_decide(llm: Dict[str, Any], vision: VisionHint) -> Dict[str, Any]:
    target_bin_llm = llm.get('target_bin', 'unknown')
    action_llm = llm.get('action', 'no_move')
    p_l = float(llm.get('confidence', 0.0))
    llm_pre = set(llm.get('preconditions', []))
    vision_pre = derive_preconditions_from_vision(vision)
    preconditions = llm_pre | vision_pre

    target_bin_v = map_vision_to_bin(vision)
    agree = target_bin_llm == target_bin_v and target_bin_llm != 'unknown'
    p_fused = fuse_confidence(vision.confidence, p_l, agree)

    if preconditions & (PRECONDITION_BLOCKLIST | OPTIONAL_BLOCKLIST):
        target = target_bin_llm if target_bin_llm != 'unknown' else target_bin_v
        return {
            'action': 'NO_MOVE_INSTRUCT',
            'target_bin': target,
            'p_fused': p_fused,
            'preconditions': sorted(preconditions),
            'speak': llm.get('speak', '전처리가 필요합니다. 라벨 제거 또는 세척 후 다시 넣어주세요.'),
            'why': 'precondition_required',
        }

    if not agree or p_fused < settings.ask_threshold:
        return {
            'action': 'NO_MOVE_ASK',
            'target_bin': 'unknown',
            'p_fused': p_fused,
            'preconditions': sorted(preconditions),
            'speak': llm.get('clarify_question') or '재질 확인이 필요합니다. 분리배출 표시를 보여주세요.',
            'why': 'low_conf_or_disagree',
        }

    if action_llm == 'move' and p_fused >= settings.move_threshold:
        return {
            'action': f"MOVE_{target_bin_llm.upper()}",
            'target_bin': target_bin_llm,
            'p_fused': p_fused,
            'preconditions': sorted(preconditions),
            'speak': llm.get('speak', ''),
            'why': 'high_conf_move',
        }

    return {
        'action': 'NO_MOVE_ASK',
        'target_bin': target_bin_llm,
        'p_fused': p_fused,
        'preconditions': sorted(preconditions),
        'speak': llm.get('clarify_question') or f'{target_bin_llm}로 배출해도 되는지 확인이 필요합니다.',
        'why': 'mid_conf_confirm',
    }
