from typing import Any

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

GEMINI_CONFIG: dict[str, Any] = {
    "vertexai": True,  # Project and location are read from GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.
    "max_output_tokens": 8192,
    "thinking_level": "low",
    "safety_settings": {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    },
}


def get_chat_model(model: str, **overrides: Any) -> ChatGoogleGenerativeAI:
    """Returns a Gemini chat model configured with the default GEMINI_CONFIG."""
    return ChatGoogleGenerativeAI(model=model, **{**GEMINI_CONFIG, **overrides})
