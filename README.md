# ActionGraph — AI-Powered Meeting Intelligence

ActionGraph is an end-to-end AI application that converts meeting audio into structured, evidence-backed action items. It combines speech-to-text transcription with LLM-based extraction and validation, exposed through a FastAPI REST API.

## Features

- **Audio transcription:** Transcribes meeting recordings using `faster-whisper`.
- **Language detection:** Returns the detected language and language-detection probability.
- **Timestamped segments:** Provides transcription segments with start and end times.
- **Action-item extraction:** Uses Gemini to identify tasks, owners, deadlines, and whether an action is explicit or implicit.
- **Evidence validation:** Checks whether each extracted supporting quote appears in the transcript.
- **Structured responses:** Uses Pydantic models to validate and serialize API responses.
- **Transcript preprocessing:** Cleans transcript text before downstream processing.
- **Evaluation workflow:** Evaluates extraction quality using semantic matching and precision, recall, and F1 metrics.

> Evidence validation checks textual presence in the transcript. It does not independently verify that an action, owner, deadline, or transcription is factually correct.

## Architecture

```text
Meeting Audio
    |
    v
FastAPI Upload Endpoint
    |
    v
faster-whisper (Speech-to-Text)
    |
    v
Transcript + Language + Timestamped Segments
    |
    v
Gemini (Action-Item Extraction)
    |
    v
Structured Action Items
    |
    v
Evidence Validation
    |
    v
Validated API Response
```

## Tech Stack

- **Language:** Python
- **API:** FastAPI, Uvicorn
- **Speech-to-text:** faster-whisper (Whisper)
- **LLM:** Google Gemini API (`google-genai`)
- **Validation and schemas:** Pydantic
- **Configuration:** pydantic-settings
- **Evaluation:** Sentence Transformers, cosine similarity

## Project Structure

```text
ActionGraph/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── config/
│   │   └── settings.py
│   ├── llm/
│   │   └── client.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── meeting_service.py
│   │   └── speech_to_text_service.py
│   ├── utils/
│   │   └── text_utils.py
│   └── main.py
├── data/
│   ├── transcripts/
│   └── evaluations/
├── scripts/
│   └── evaluate.py
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+ (use the version supported by your installed dependencies)
- A Google Gemini API key
- Git

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ActionGraph
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
WHISPER_MODEL=base
```

Do not commit `.env` or expose your API key. Keep `.env.example` free of secrets.

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

- API root: `http://127.0.0.1:8000`
- Interactive Swagger docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/transcripts/preprocess` | Clean and preprocess transcript text |
| `POST` | `/meetings/extract-actions` | Extract action items from transcript text |
| `POST` | `/meetings/validate-actions` | Validate extracted action-item evidence |
| `POST` | `/transcribe` | Transcribe an uploaded audio file |
| `POST` | `/meetings/process-audio` | Transcribe audio, extract action items, and validate evidence |

Use `/docs` to inspect each endpoint's request schema and try it interactively.

### Process an audio meeting

Send a multipart file upload to:

```http
POST /meetings/process-audio
```

Supported audio extensions in the current implementation:

`.wav`, `.mp3`, `.m4a`, `.mp4`, `.webm`, `.ogg`, `.flac`

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/meetings/process-audio" \
  -F "file=@meeting.wav"
```

Example response shape:

```json
{
  "transcript": "Priya, I'll take care of the authentication issue.",
  "language": "en",
  "language_probability": 0.94,
  "action_items": [
    {
      "action": "Take care of the authentication issue",
      "owner": "Priya",
      "deadline": null,
      "type": "explicit",
      "confidence": 0.95,
      "evidence": "Priya, I'll take care of the authentication issue.",
      "evidence_valid": true,
      "validation_reason": "Evidence exists in the transcript."
    }
  ]
}
```

The values above are illustrative; actual output depends on the recording and model responses.

## Evaluation

The project includes an evaluation script and a small labeled dataset for measuring action-item extraction quality. The workflow uses semantic similarity to match predicted action descriptions against expected actions and reports precision, recall, and F1, with additional action-type metrics.

Evaluation results depend on dataset size, annotation quality, matching threshold, and model behavior. Scores on a small dataset should not be interpreted as a guarantee of production performance.

Run the evaluation script:

```bash
python scripts/evaluate.py
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `GEMINI_MODEL` | Gemini model identifier | `gemini-2.5-flash` |
| `WHISPER_MODEL` | faster-whisper model size | `base` |

For lower-memory deployments, a smaller Whisper model such as `tiny` may reduce resource usage at the cost of transcription quality.

## Deployment

The API can be deployed to a Python-compatible hosting platform such as Render.

Typical Web Service settings:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment variables:** `GEMINI_API_KEY`, `GEMINI_MODEL`, `WHISPER_MODEL`

Set secrets through the hosting provider's environment-variable settings, not in source control. Check the hosting provider's current resource limits before deploying Whisper; model downloads, startup time, and memory usage may exceed the limits of small instances.

## Limitations and Future Improvements

- Transcription mistakes can propagate into action-item extraction.
- Evidence matching verifies quote presence, not semantic correctness.
- Extraction quality depends on audio quality, transcript clarity, and LLM behavior.
- The current evaluation dataset is small and should be expanded for more reliable assessment.
- Production hardening could include authentication, upload-size limits, rate limiting, structured logging, monitoring, and stronger semantic validation.

## License

Add a license file and update this section if you intend to distribute the project publicly.
