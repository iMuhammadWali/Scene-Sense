"""Request/response schemas for SceneSense APIs."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    hardware: dict[str, Any]
    models: dict[str, Any]
    config: dict[str, Any]


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    size_bytes: int
    status: str
    message: str


class StatusResponse(BaseModel):
    session_id: str
    status: str
    stage: Optional[str] = None
    progress: float = 0.0
    message: Optional[str] = None
    video_ready: bool = False
    stages: list[dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class FrameScore(BaseModel):
    rank: int
    frame_id: int
    timestamp: float
    similarity: float
    faiss_index: Optional[int] = None


class TemporalGroupSchema(BaseModel):
    group_id: int
    start_timestamp: float
    end_timestamp: float
    frames: list[FrameScore]


class EvidenceFrame(BaseModel):
    frame_id: Optional[int] = None
    timestamp: float
    similarity: Optional[float] = None
    source: Optional[str] = None


class QuestionResponse(BaseModel):
    question_id: str
    query: str
    answer: str
    query_type: Literal["SIMPLE", "COMPLEX"]
    path: Literal["FAST", "SLOW"]
    vlm: str
    evidence: list[EvidenceFrame]
    retrieval: dict[str, Any]
    timings: dict[str, Any]
    debug: dict[str, Any]


class MetadataResponse(BaseModel):
    session_id: str
    metadata: dict[str, Any]
    visual_memory_summary: dict[str, Any]
    video_ready: bool
