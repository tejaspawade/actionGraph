from fastapi import APIRouter
from app.models.schemas import (
    TranscriptRequest,
    TranscriptResponse,
    ActionItemResponse,
    ValidatedActionItemResponse
)
from app.services.meeting_service import MeetingService

import os
import tempfile

from fastapi import File, HTTPException, UploadFile

from app.models.schemas import SpeechToTextResponse, AudioMeetingResponse
from app.services.speech_to_text_service import (
    get_speech_to_text_service,
)



router = APIRouter()

meeting_service = MeetingService()


@router.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@router.post(
    "/transcripts/preprocess",
    response_model=TranscriptResponse
)
def preprocess_transcript(request: TranscriptRequest):

    cleaned_transcript = meeting_service.preprocess_transcript(
        request.transcript
    )

    return TranscriptResponse(
        original_length=len(request.transcript),
        cleaned_length=len(cleaned_transcript),
        transcript=cleaned_transcript
    )


@router.post("/meetings/extract-actions", response_model=ActionItemResponse)
def extract_action_items(request: TranscriptRequest):

   return meeting_service.extract_action_items(
        request.transcript
    )

@router.post(
    "/meetings/validate-actions",
    response_model=ValidatedActionItemResponse
)
def validate_action_items(
    request: TranscriptRequest
):

    extracted_actions = meeting_service.extract_action_items(
        request.transcript
    )

    return meeting_service.validate_action_items(
        request.transcript,
        extracted_actions
    )

@router.post(
    "/transcribe",
    response_model=SpeechToTextResponse,
)
def transcribe_audio(file: UploadFile = File(...)):
    allowed_extensions = {
        ".wav", ".mp3", ".m4a", ".mp4",
        ".webm", ".ogg", ".flac",
    }

    extension = os.path.splitext(file.filename or "")[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio file extension.",
        )

    temp_path = None

    try:
        # Save the upload to a temporary file.
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

            while chunk := file.file.read(1024 * 1024):
                temp_file.write(chunk)

        result = get_speech_to_text_service().transcribe_file(
            temp_path
        )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Audio transcription failed: {exc}",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        file.file.close()



@router.post(
    "/meetings/process-audio",
    response_model=AudioMeetingResponse,
)
def process_meeting_audio(file: UploadFile = File(...)):
    allowed_extensions = {
        ".wav", ".mp3", ".m4a", ".mp4",
        ".webm", ".ogg", ".flac",
    }

    extension = os.path.splitext(file.filename or "")[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio file extension.",
        )

    temp_path = None

    try:
        # 1. Save the uploaded audio temporarily.
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

            while chunk := file.file.read(1024 * 1024):
                temp_file.write(chunk)

        # 2. Convert speech into text.
        speech_service = get_speech_to_text_service()
        transcription = speech_service.transcribe_file(temp_path)

        transcript = transcription["transcript"].strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="No speech could be transcribed from the audio.",
            )

        # 3. Extract action items using your existing Gemini service.
        meeting_service = MeetingService()
        extracted = meeting_service.extract_action_items(transcript)

        # 4. Validate the evidence against the transcript.
        validated = meeting_service.validate_action_items(
            transcript=transcript,
            action_items=extracted,
        )

        # 5. Return the complete result.
        return AudioMeetingResponse(
            transcript=transcript,
            language=transcription["language"],
            language_probability=transcription[
                "language_probability"
            ],
            action_items=validated.action_items,
        )

    except HTTPException:
        raise

    except Exception as exc:
        # Log the exception in a production application.
        raise HTTPException(
            status_code=500,
            detail=f"Meeting audio processing failed: {exc}",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        file.file.close()
