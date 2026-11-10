"""Submission models with float fields, for testing non-finite handling.

DivergenceAwareSubmission shows the pattern for letting an agent say "this
quantity diverges" without sending NaN or infinity: an explicit flag, and the
number left empty when the flag is set.
"""

from pydantic import BaseModel, Field, FiniteFloat, model_validator


class ValueSubmission(BaseModel):
    value: float
    samples: list[float] = Field(default_factory=list)


class DivergenceAwareSubmission(BaseModel):
    is_finite: bool = Field(description="False if the quantity diverges or is undefined.")
    value: FiniteFloat | None = Field(default=None, description="The value; leave empty if is_finite is false.")

    @model_validator(mode="after")
    def value_matches_flag(self):
        if self.is_finite and self.value is None:
            raise ValueError("value is required when is_finite is true")
        if not self.is_finite and self.value is not None:
            raise ValueError("leave value empty when is_finite is false")
        return self
