from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt


class InteractionType(StrEnum):
    VIEW = "VIEW"
    APPLY = "APPLY"
    FINISHED = "FINISHED"
    USER_CANCEL = "USER_CANCEL"
    SYSTEM_CANCEL = "SYSTEM_CANCEL"


class UserDTO(BaseModel):
    id: str = Field(min_length=1)
    location_id: str = Field(min_length=1)
    is_strict_location: bool
    has_mk: bool


class ShiftDTO(BaseModel):
    id: str = Field(min_length=1)
    start_at: datetime
    location_id: str = Field(min_length=1)
    task_type: str = Field(min_length=1)
    employer_id: str = Field(min_length=1)
    workplace_id: str = Field(min_length=1)
    need_mk: bool
    id_differential: bool
    hours: PositiveInt
    reward: NonNegativeFloat
    capacity: PositiveInt


class EventDTO(BaseModel):
    id: UUID
    shift_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    interaction: InteractionType
    ts: datetime


class BatchUsersRequest(BaseModel):
    items: list[UserDTO]


class BatchShiftsRequest(BaseModel):
    items: list[ShiftDTO]


class BatchEventsRequest(BaseModel):
    items: list[EventDTO]


class BatchWriteResponse(BaseModel):
    accepted: int


class CountResponse(BaseModel):
    count: int


class PrepareResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool


class PredictResponse(BaseModel):
    user_ids: list[str]


class PredictRequest(BaseModel):
    shift: ShiftDTO
    limit: int = Field(default=10, ge=1, le=100)
