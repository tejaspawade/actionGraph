from typing import Literal
from pydantic import BaseModel, Field


class TranscriptRequest(BaseModel):
    transcript: str = Field(
        ...,
        min_length=10,
        description="Raw meeting transcript"
    )


class TranscriptResponse(BaseModel):
    original_length: int
    cleaned_length: int
    transcript: str


class ActionItem(BaseModel):
    action: str = Field(
        description="The task that needs to be performed"
    )

    owner: str | None = Field(
        default=None,
        description="Person responsible for the action"
    )

    deadline: str | None = Field(
        default=None,
        description="Deadline mentioned in the transcript"
    )

    type: Literal["explicit", "implicit"] = Field(
        description="Whether the commitment is explicit or implicit"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that this is a valid action item"
    )

    evidence: str = Field(
        description="Exact transcript evidence supporting the action"
    )

class ValidatedActionItem(ActionItem):
    evidence_valid: bool = Field(
        description="Whether the evidence exists in the transcript"
    )

    validation_reason: str = Field(
        description="Explanation of the evidence validation result"
    )


class ActionItemResponse(BaseModel):
    action_items: list[ActionItem]

class ValidatedActionItemResponse(BaseModel):
    action_items: list[ValidatedActionItem]

class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class SpeechToTextResponse(BaseModel):
    transcript: str
    language: str
    language_probability: float = Field(ge=0.0, le=1.0)
    segments: list[TranscriptSegment]

class AudioMeetingResponse(BaseModel):
    transcript: str
    language: str
    language_probability: float
    action_items: list[ValidatedActionItem]