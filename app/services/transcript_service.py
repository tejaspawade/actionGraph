from pathlib import Path
from app.utils.text_utils import clean_transcript


class TranscriptService:

    def load_from_file(self, file_path: str) -> str:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Transcript file not found: {file_path}"
            )

        if path.suffix.lower() != ".txt":
            raise ValueError(
                "Only .txt transcript files are currently supported."
            )

        transcript = path.read_text(
            encoding="utf-8"
        )

        return clean_transcript(transcript)