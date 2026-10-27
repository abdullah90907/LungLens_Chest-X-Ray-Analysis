"""Typed frontend contract, also exposed through FastAPI's OpenAPI document."""
from typing import Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    label: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False, description='Uncalibrated sigmoid model score; not a diagnostic probability')
    reference_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    above_reference_threshold: bool


class ImageInfo(BaseModel):
    width: int
    height: int
    view: Literal['AP', 'PA'] | None = None
    modality_verified: bool = False


class ExplanationItem(BaseModel):
    label: str
    explanation: str


class Explanation(BaseModel):
    status: Literal['not_requested', 'disabled', 'not_generated', 'unavailable', 'generated']
    detail: str | None = None
    model: str | None = None
    items: list[ExplanationItem] | None = None
    review_required: bool | None = None
    notice: str | None = None


class AnalysisResponse(BaseModel):
    analysis_id: str
    created_at: str
    project: Literal['LungLens']
    task: Literal['chest_xray_findings']
    model: dict
    image: ImageInfo
    preprocessing: str
    score_type: Literal['uncalibrated_sigmoid']
    threshold_policy: str
    findings: list[Finding] = Field(min_length=18, max_length=18)
    notice: str
    limitations: list[str]
    explanation: Explanation
    input_metadata: dict | None = None
    quality: dict | None = None


class HeatmapResponse(BaseModel):
    method: str
    finding: str
    score: float = Field(ge=0,le=1,allow_inf_nan=False)
    view: Literal['AP','PA'] | None = None
    score_type: Literal['uncalibrated_sigmoid']
    status: Literal['generated','no_positive_attribution']
    target: str
    model: dict
    width: int
    height: int
    crop_box: list[int]
    preview_crop_box: list[int]
    heatmap_png_base64: str
    overlay_png_base64: str
    coordinates: str
    notice: str


class AnatomyStructure(BaseModel):
    label: str
    color_rgb: list[int]
    mask_png_base64: str
    crop_area_fraction: float = Field(ge=0,le=1,allow_inf_nan=False)


class AnatomyResponse(BaseModel):
    method: str
    model: dict
    threshold: float
    width: int
    height: int
    crop_box: list[int]
    preview_crop_box: list[int]
    structures: list[AnatomyStructure] = Field(min_length=3,max_length=3)
    overlay_png_base64: str
    notice: str


class EvaluationObservation(BaseModel):
    case_id: str = Field(min_length=1,max_length=80,pattern=r'^[A-Za-z0-9_-]+$')
    label: str = Field(min_length=1,max_length=80)
    truth: int = Field(strict=True,ge=0,le=1)
    score: float = Field(ge=0,le=1,allow_inf_nan=False)


class EvaluationRequest(BaseModel):
    model_weights: Literal['densenet121-res224-all']
    checkpoint_sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    observations: list[EvaluationObservation] = Field(min_length=1,max_length=50000)
