import json
import re

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a financial expert specializing in market sentiment analysis. "
    "Analyze the sentiment of the provided financial text and respond ONLY with "
    "a valid JSON object in this exact format: "
    '{"sentiment": "Positive" | "Negative" | "Neutral", "confidence_score": <float between 0 and 1>}. '
    "Do not include any explanation, markdown, or extra text — just the raw JSON object."
)


def build_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="lm-studio")


def analyze(text: str, client: OpenAI, model: str) -> dict:
    """
    Run sentiment analysis on text via local LM Studio.

    Returns:
        {"sentiment": "Positive"|"Negative"|"Neutral", "confidence_score": float}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze the sentiment of this financial text:\n\n{text}"},
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content
    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model response: {raw}")
    return json.loads(match.group())
