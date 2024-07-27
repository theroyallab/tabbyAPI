from pydantic import BaseModel


class CurrentModelResponse(BaseModel):
    result: str


class MaxLengthResponse(BaseModel):
    value: str
