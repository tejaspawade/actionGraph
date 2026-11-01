from dataclasses import dataclass


@dataclass
class EvidenceValidationResult:
    valid: bool
    reason: str


def normalize_text(text: str) -> str:
    """
    Normalize text before comparison.
    """

    return " ".join(text.lower().split())


def validate_evidence(
    evidence: str,
    transcript: str
) -> EvidenceValidationResult:

    if not evidence:
        return EvidenceValidationResult(
            valid=False,
            reason="Evidence is empty."
        )

    normalized_evidence = normalize_text(evidence)
    normalized_transcript = normalize_text(transcript)

    if normalized_evidence in normalized_transcript:
        return EvidenceValidationResult(
            valid=True,
            reason="Evidence exists in the transcript."
        )

    return EvidenceValidationResult(
        valid=False,
        reason="Evidence was not found in the transcript."
    )