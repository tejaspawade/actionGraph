
from app.llm.client import GeminiClient

from app.models.schemas import (
    ActionItemResponse,
    ValidatedActionItem,
    ValidatedActionItemResponse,
)

from app.utils.text_utils import clean_transcript
from app.utils.evidence_validator import validate_evidence


class MeetingService:

    def __init__(self):
        self.llm = GeminiClient()

    def preprocess_transcript(
        self,
        transcript: str
    ) -> str:
        return clean_transcript(transcript)

    def extract_action_items(
        self,
        transcript: str
    ) -> ActionItemResponse:

        cleaned_transcript = self.preprocess_transcript(
            transcript
        )

        prompt = f"""
You are an AI assistant specialized in extracting
accurate action items from meeting transcripts.

Your task is to identify genuine tasks that participants
have committed to performing or that are reasonably
implied by the conversation.

ACTION ITEM RULES

1. Extract only actionable tasks supported by the transcript.

2. Do not treat every question, problem, suggestion,
   plan, or discussion topic as an action item.

3. Do not create an action item solely because participants
   mention an issue that needs attention.

4. Do not interpret a planned discussion or meeting as a
   commitment to implement, fix, investigate, or complete
   the topic being discussed.

5. Do not invent tasks, owners, deadlines, or commitments.

ACTION TYPE CLASSIFICATION

Explicit:
Use "explicit" when a participant clearly commits to a
specific task.

Examples:
- "I'll fix the authentication issue."
- "I'll send the report tomorrow."
- "I will investigate the failed payments."

Implicit:
Use "implicit" only when the transcript provides sufficient
evidence of a specific intended task, but the task or
commitment requires reasonable interpretation.

Examples:
- "The dashboard is slow; I'll look into it."
  Action: Investigate the dashboard performance issue.
  Type: explicit, because the speaker directly commits
  to looking into the problem.

- "Could someone investigate the failed payments?"
  This is a request, not proof that anyone accepted
  responsibility. Do not invent an owner or claim that
  the task was accepted.

Important:
An action is not automatically implicit just because its
wording differs from the transcript. Classify it according
to the strength and nature of the commitment.

DISCUSSION TOPICS

Do not extract an implementation task from statements like:
- "Let's discuss the database migration tomorrow."
- "We should review the release plan next Monday."
- "We need to decide on an approach."

These are discussion or decision-planning statements.
Do not turn them into implementation tasks unless the
transcript also establishes a specific actionable commitment.

OWNERS, DEADLINES, AND EVIDENCE

- Set owner only when the responsible participant is
  supported by the transcript; otherwise use null.
- Set deadline only when a deadline is stated; otherwise
  use null.
- Evidence must be an exact quote from the cleaned transcript.
- Every extracted action must have evidence supporting it.
- Do not fabricate or paraphrase the evidence.
- Confidence must be between 0 and 1.
- Return an empty action_items list if no genuine action
  items are present.
- Avoid duplicate action items describing the same task.

Review the complete transcript before extracting actions.
Return the result using the required structured schema.

MEETING TRANSCRIPT:

{cleaned_transcript}
"""

        return self.llm.generate_structured(
            prompt=prompt,
            response_schema=ActionItemResponse,
        )

    def validate_action_items(
        self,
        transcript: str,
        action_items: ActionItemResponse
    ) -> ValidatedActionItemResponse:

        cleaned_transcript = self.preprocess_transcript(
            transcript
        )

        validated_items = []

        for item in action_items.action_items:

            validation = validate_evidence(
                evidence=item.evidence,
                transcript=cleaned_transcript,
            )

            validated_item = ValidatedActionItem(
                action=item.action,
                owner=item.owner,
                deadline=item.deadline,
                type=item.type,
                confidence=item.confidence,
                evidence=item.evidence,
                evidence_valid=validation.valid,
                validation_reason=validation.reason,
            )

            validated_items.append(validated_item)

        return ValidatedActionItemResponse(
            action_items=validated_items
        )
