"""Schema for retrieval and answer-quality golden questions."""

from pydantic import BaseModel, Field


class GoldenQuestion(BaseModel):
    """A hand-authored evaluation question and expected behavior."""

    id: str
    question: str
    expected_doc_ids: list[str]
    expected_answer_summary: str
    expected_answerable: bool
    category: str = Field(min_length=1)
