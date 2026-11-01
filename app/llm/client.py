from typing import Type, TypeVar
from google import genai
from pydantic import BaseModel
from app.config.settings import settings


T = TypeVar("T", bound=BaseModel)


class GeminiClient:

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

        self.model = settings.gemini_model

    def generate(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return response.text

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T]
    ) -> T:

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }
        )

        return response.parsed