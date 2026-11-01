
from functools import lru_cache
from pathlib import Path
import os
from faster_whisper import WhisperModel


class SpeechToTextService:

    def __init__(self):
        # Runs locally on the CPU.
        # The model is downloaded automatically on first use.
        model_name = os.getenv("WHISPER_MODEL", "base")
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
        )

    def transcribe_file(self, audio_path: str) -> dict:
        path = Path(audio_path)

        if not path.exists() or not path.is_file():
            raise FileNotFoundError("Audio file not found.")

        segments, info = self.model.transcribe(
            str(path),
            beam_size=5,
            vad_filter=True,
        )

        # Transcription starts when the generator is iterated.
        transcript_segments = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                transcript_segments.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text,
                })

        transcript = " ".join(
            segment["text"] for segment in transcript_segments
        )

        return {
            "transcript": transcript,
            "language": info.language,
            "language_probability": round(
                info.language_probability, 4
            ),
            "segments": transcript_segments,
        }


@lru_cache(maxsize=1)
def get_speech_to_text_service() -> SpeechToTextService:
    # Reuse the model instead of loading it for every request.
    return SpeechToTextService()
