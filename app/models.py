"""
Pydantic models for request / response / error payloads.

Matches the Nebius Academy assignment spec exactly:
  Request:  {"github_url": "..."}
  Response: {"summary": "...", "technologies": [...], "structure": "..."}
  Error:    {"status": "error", "message": "..."}
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    github_url: str = Field(..., description="URL of a public GitHub repository")


class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="Human-readable description of the project")
    technologies: list[str] = Field(
        ..., description="Main technologies, languages, and frameworks used"
    )
    structure: str = Field(
        ..., description="Brief description of the project structure"
    )


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
