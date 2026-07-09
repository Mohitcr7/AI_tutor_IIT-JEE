"""
LLM client for the tutor, backed by the Google Gemini API.

A single structured-output call returns the concept explanation, common traps,
and step-by-step worked examples together. Gemini's `response_schema` validates
the output against a Pydantic model, so downstream code gets typed objects
rather than free-form text to parse.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment. Get a free key
at https://aistudio.google.com/apikey. The client is created lazily so importing
this module (e.g. during the store build) does not require a key.
"""

import os
from typing import List, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

# Gemini 2.5 Flash-Lite: fastest tier, most generous free quota. Step up via
# TUTOR_MODEL (e.g. gemini-2.5-flash / gemini-2.5-pro) for deeper reasoning.
DEFAULT_MODEL = os.environ.get("TUTOR_MODEL", "gemini-2.5-flash-lite")

SYSTEM_PROMPT = (
    "You are an expert IIT-JEE Advanced tutor for Physics, Chemistry, and "
    "Mathematics. You explain the reasoning behind concepts the way a top "
    "coaching instructor would: precise, exam-focused, and honest about where "
    "students go wrong.\n\n"
    "You never invent past-year questions — you only work with the questions "
    "provided to you. Each question comes with an official, verified solution. "
    "When you teach a question, you TEACH THROUGH the provided solution: you "
    "explain its steps and the reasoning behind them. You do NOT re-derive the "
    "answer yourself or change the method, because the provided solution is "
    "ground truth. If a provided solution appears to be wrong, say so plainly "
    "rather than silently correcting it."
)


class Trap(BaseModel):
    trap: str
    why_students_fall_for_it: str
    correction: str


class WorkedExample(BaseModel):
    qid: str                # exact id of the question being taught
    what_it_tests: str      # one line: why this question reinforces the concept
    approach: str           # the first insight / how to start — a hint, not the full solution
    steps: List[str]        # step-by-step walkthrough of the verified solution, with the "why" at each step
    trap: str               # the specific mistake to avoid on THIS problem


class TutorResponse(BaseModel):
    concept_explanation: str
    common_traps: List[Trap]
    worked_examples: List[WorkedExample]


_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it (or add it to a .env file) "
                "before running the tutor. Get a free key at "
                "https://aistudio.google.com/apikey. See .env.example."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def generate_tutor_response(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    model: Optional[str] = None,
) -> TutorResponse:
    """
    Run one structured Gemini call and return a validated TutorResponse.

    Raises RuntimeError if the model returns nothing parseable (e.g. the output
    was truncated or blocked by a safety filter).

    Thinking budget and max output tokens are left at the model default so the
    full output budget is available for the JSON — the reliable choice for
    structured output across models (an explicit low cap risks truncating the
    response and yielding an unparseable result).
    """
    client = _get_client()
    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=TutorResponse,
        ),
    )

    parsed = response.parsed
    if parsed is None:
        raise RuntimeError(
            "Model did not return a valid tutor response (the output may have "
            "been truncated or blocked by a safety filter). Raw output: "
            f"{(response.text or '')[:300]}"
        )
    return parsed
