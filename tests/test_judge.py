"""Tests for LLM-as-judge citation verification."""

from app.generation.generator import Citation, GeneratedAnswer
from app.llm.client import LLMClient
from app.retrieval.base import RetrievedChunk
from app.verification.cache import JudgeCache, build_judge_cache_key
from app.verification.judge import CitationVerifier, _extract_json_object, _parse_judge_output
from app.verification.schemas import Verdict


class FakeJudgeLLM(LLMClient):
    """Deterministic judge client for tests."""

    def __init__(self, responses: list[str]) -> None:
        """Initialize with queued responses."""
        self.responses = responses
        self.prompts: list[str] = []

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        """Return the next queued response."""
        self.prompts.append(user_prompt)
        return self.responses.pop(0)


def _citation(chunk_id: str) -> Citation:
    """Create a citation fixture."""
    return Citation(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_path=f"{chunk_id}.md",
        quoted_text="source quote",
    )


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    """Create a retrieved chunk fixture."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_path=f"{chunk_id}.md",
        text=text,
        section="Section",
        score=1.0,
        rank=1,
        retriever_name="hybrid",
    )


def test_split_into_claims_handles_multiple_sentences_and_markers() -> None:
    """Assert claims are associated with citation markers in sentence order."""
    verifier = CitationVerifier(llm_client=FakeJudgeLLM([]))
    answer = (
        "Customers can reset passwords from sign-in [a]. "
        "The link expires after thirty minutes [a] and can be used once [b]. "
        "This uncited sentence is ignored."
    )

    pairs = verifier._split_into_claims(answer, [_citation("a"), _citation("b")])

    assert len(pairs) == 3
    assert pairs[0][0] == "Customers can reset passwords from sign-in ."
    assert pairs[0][1].chunk_id == "a"
    assert pairs[1][1].chunk_id == "a"
    assert pairs[2][1].chunk_id == "b"


def test_split_into_claims_does_not_split_inside_email_addresses() -> None:
    """Assert email domains do not create bogus partial claims."""
    verifier = CitationVerifier(llm_client=FakeJudgeLLM([]))
    answer = (
        "Allow messages from notifications@northstardesk.example before requesting a new "
        "link [a]."
    )

    pairs = verifier._split_into_claims(answer, [_citation("a")])

    assert len(pairs) == 1
    assert pairs[0][0] == (
        "Allow messages from notifications@northstardesk.example before requesting a new link ."
    )


def test_extract_json_object_handles_fences_and_trailing_text() -> None:
    """Assert judge JSON extraction handles common model formatting."""
    fenced = '```json\n{"verdict": "SUPPORTED", "reasoning": "ok"}\n```'
    trailing = '{"verdict": "UNSUPPORTED", "reasoning": "bad"}\nExtra commentary.'

    assert _extract_json_object(fenced)["verdict"] == "SUPPORTED"
    assert _extract_json_object(trailing)["verdict"] == "UNSUPPORTED"


def test_parse_judge_output_falls_back_to_unsupported_on_malformed_json() -> None:
    """Assert malformed judge output becomes an unsupported verdict."""
    verdict = _parse_judge_output(
        raw_output="not json",
        chunk_id="a",
        claim_excerpt="claim",
    )

    assert verdict.verdict == Verdict.UNSUPPORTED
    assert "could not be parsed" in verdict.judge_reasoning


def test_verify_populates_verified_answer_and_marks_flagged_citations() -> None:
    """Assert verifier separates supported and unsupported citations."""
    llm = FakeJudgeLLM(
        [
            '{"verdict": "SUPPORTED", "reasoning": "The source supports it."}',
            '{"verdict": "UNSUPPORTED", "reasoning": "The source does not say this."}',
        ]
    )
    verifier = CitationVerifier(llm_client=llm)
    generated = GeneratedAnswer(
        answer_text="Reset links expire after thirty minutes [a]. Refunds are always instant [b].",
        citations=[_citation("a"), _citation("b")],
        raw_llm_output="raw",
    )
    chunks = [
        _chunk("a", "The link expires after thirty minutes and can be used only once."),
        _chunk("b", "Removing seats does not usually create an immediate refund."),
    ]

    verified = verifier.verify(generated=generated, retrieved_chunks=chunks)

    assert verified.all_supported is False
    assert [citation.chunk_id for citation in verified.verified_citations] == ["a"]
    assert [citation.chunk_id for citation in verified.flagged_citations] == ["b"]
    assert "[b]" not in verified.answer_text
    assert "[unverified]" in verified.answer_text
    assert [verdict.verdict for verdict in verified.verdicts] == [
        Verdict.SUPPORTED,
        Verdict.UNSUPPORTED,
    ]


def test_verify_uses_cached_judge_result_on_cache_hit(tmp_path) -> None:
    """Assert cached judge results avoid repeat LLM calls."""
    chunk = _chunk("a", "The reset link expires after thirty minutes.")
    claim_excerpt = "The reset link expires after thirty minutes ."
    cache = JudgeCache(tmp_path / "judge_cache.sqlite3")
    cache_key = build_judge_cache_key(
        claim_excerpt=claim_excerpt,
        chunk_id="a",
        source_text=chunk.text,
    )
    cache.set(
        cache_key=cache_key,
        verdict=_parse_judge_output(
            raw_output='{"verdict": "SUPPORTED", "reasoning": "Cached support."}',
            chunk_id="a",
            claim_excerpt=claim_excerpt,
        ),
        model_name="fake-model",
        prompt_version="test-v1",
    )
    llm = FakeJudgeLLM([])
    verifier = CitationVerifier(
        llm_client=llm,
        cache=cache,
        enable_cache=True,
        model_name="fake-model",
        prompt_version="test-v1",
    )
    generated = GeneratedAnswer(
        answer_text="The reset link expires after thirty minutes [a].",
        citations=[_citation("a")],
        raw_llm_output="raw",
    )

    verified = verifier.verify(generated=generated, retrieved_chunks=[chunk])

    assert len(llm.prompts) == 0
    assert verified.verdicts[0].verdict == Verdict.SUPPORTED
    assert verified.verdicts[0].judge_reasoning == "Cached support."


def test_verify_stores_judge_result_on_cache_miss(tmp_path) -> None:
    """Assert cache misses call the LLM and persist the parsed verdict."""
    chunk = _chunk("a", "The reset link expires after thirty minutes.")
    llm = FakeJudgeLLM(['{"verdict": "SUPPORTED", "reasoning": "The source says exactly this."}'])
    cache = JudgeCache(tmp_path / "judge_cache.sqlite3")
    verifier = CitationVerifier(
        llm_client=llm,
        cache=cache,
        enable_cache=True,
        model_name="fake-model",
        prompt_version="test-v1",
    )
    generated = GeneratedAnswer(
        answer_text="The reset link expires after thirty minutes [a].",
        citations=[_citation("a")],
        raw_llm_output="raw",
    )

    verified = verifier.verify(generated=generated, retrieved_chunks=[chunk])
    cached = cache.get(
        build_judge_cache_key(
            claim_excerpt=verified.verdicts[0].claim_excerpt,
            chunk_id="a",
            source_text=chunk.text,
        )
    )

    assert len(llm.prompts) == 1
    assert cached is not None
    assert cached.verdict == Verdict.SUPPORTED
    assert cached.reasoning == "The source says exactly this."
    assert cached.model_name == "fake-model"
    assert cached.prompt_version == "test-v1"
