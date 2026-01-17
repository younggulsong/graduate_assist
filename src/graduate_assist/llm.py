from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional


def generate_text(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Generate text via configured LLM provider or return None if unavailable."""
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model = os.getenv("LLM_MODEL", "gpt-4.1").strip()

    if not provider:
        return None

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.output_text
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    if provider == "ollama":
        try:
            prompt = f"{system_prompt}\n\n{user_prompt}".strip()
            payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("response")
        except Exception:
            return None

    return None
