"""Submission model for the toy experiment."""

from pydantic import BaseModel, Field


class SumSubmission(BaseModel):
    total: int = Field(description="The sum of the integers in the task.")
    method_notes: str = Field(description="One or two sentences on how the sum was computed and checked.")
