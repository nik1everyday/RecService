import typing as tp
from typing import List

from pydantic import BaseModel


class Error(BaseModel):
    error_key: str
    error_message: str
    error_loc: tp.Optional[tp.Any] = None


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class ExplainResponse(BaseModel):
    p: int
    explanation: str
