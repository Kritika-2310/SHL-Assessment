"""
Conversational agent for SHL assessment recommendation.
Uses Groq (free tier) with RAG over the FAISS catalog index.
"""
import json
import os
import pickle
import re
from pathlib import Path

import httpx

from app.index.retriever import search_catalog

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL        = "llama-3.3-70b-versatile"
MAX_TOKENS   = 2048

SYSTEM_PROMPT = """You are an SHL assessment advisor. Your only job is to help hiring managers find the right SHL Individual Test Solutions.

STRICT RULES:
1. Only discuss SHL assessments. Refuse all off-topic requests.
2. Never recommend assessments not in the catalog data provided to you.
3. Never hallucinate URLs. Only use URLs exactly as provided in catalog results.
4. Do NOT recommend on the first turn if the query is vague. Ask clarifying questions first.
5. Once you have role, level, and what to measure — provide 1-10 recommendations.
6. Honor refinements mid-conversation.
7. For comparisons, use only catalog data provided.

CLARIFY BEFORE RECOMMENDING:
- Job role / function
- Seniority level (entry, mid, senior, manager, executive)  
- What to measure (cognitive, personality, technical skills, situational judgment)

OUTPUT FORMAT — YOU MUST ALWAYS END YOUR RESPONSE WITH THIS EXACT JSON BLOCK:
```json
{
  "reply": "your conversational response here",
  "recommendations": [],
  "end_of_conversation": false
}
```

RECOMMENDATIONS FORMAT when you have enough context:
```json
{
  "reply": "Here are the assessments I recommend...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/products/product-catalog/view/java-8-new/", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

Test type abbreviations: A=Ability & Aptitude, B=Biodata & Situational Judgement, C=Competencies, D=Development & 360, E=Assessment Exercises, K=Knowledge & Skills, P=Personality & Behavior, S=Simulations

IMPORTANT: The JSON block is MANDATORY in every single response. Never skip it."""


def _format_catalog_context(entries: list[dict]) -> str:
    if not entries:
        return "No relevant catalog entries found."
    lines = ["## Relevant SHL Catalog Entries\n"]
    for e in entries:
        keys_str   = ", ".join(e.get("keys", []))
        levels_str = ", ".join(e.get("job_levels", []))
        lines.append(
            f"**{e['name']}**\n"
            f"- URL: {e['link']}\n"
            f"- Type: {keys_str}\n"
            f"- Job Levels: {levels_str or 'All levels'}\n"
            f"- Duration: {e.get('duration') or 'Not specified'}\n"
            f"- Remote: {e.get('remote', 'unknown')} | Adaptive: {e.get('adaptive', 'unknown')}\n"
            f"- Description: {(e.get('description') or '').strip()[:300]}\n"
        )
    return "\n".join(lines)


def _build_search_query(messages: list[dict]) -> str:
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_msgs[-3:])


def _extract_json_response(text: str) -> dict:
    # Try ```json ... ``` block
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try last { ... } block in text
    matches = list(re.finditer(r'\{[^{}]*"reply"[^{}]*\}', text, re.DOTALL))
    if matches:
        try:
            return json.loads(matches[-1].group(0))
        except json.JSONDecodeError:
            pass

    # Fallback
    return {"reply": text.strip(), "recommendations": [], "end_of_conversation": False}


def _validate_recommendations(recs: list, all_entries: list[dict]) -> list:
    valid_urls  = {e["link"] for e in all_entries}
    valid_map   = {e["name"]: e for e in all_entries}
    cleaned = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        if r.get("url") in valid_urls:
            cleaned.append(r)
        elif r.get("name") in valid_map:
            entry = valid_map[r["name"]]
            cleaned.append({
                "name":      entry["name"],
                "url":       entry["link"],
                "test_type": r.get("test_type", ", ".join(entry.get("keys", []))),
            })
    return cleaned


def run_agent(messages: list[dict]) -> dict:
    api_key = os.environ.get("GROQ_API_KEY", "")

    # Retrieve relevant catalog entries
    query        = _build_search_query(messages)
    catalog_hits = search_catalog(query, top_k=15)
    catalog_ctx  = _format_catalog_context(catalog_hits)

    # Augment last user message with catalog context
    augmented = []
    for i, msg in enumerate(messages):
        if i == len(messages) - 1 and msg["role"] == "user":
            augmented.append({
                "role": "user",
                "content": (
                    f"{msg['content']}\n\n"
                    f"---\n{catalog_ctx}\n---\n\n"
                    "Use ONLY the catalog entries above. "
                    "You MUST end your response with the JSON block."
                ),
            })
        else:
            augmented.append({"role": msg["role"], "content": msg["content"]})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       MODEL,
        "max_tokens":  MAX_TOKENS,
        "temperature": 0.1,
        "messages":    [{"role": "system", "content": SYSTEM_PROMPT}] + augmented,
    }

    try:
        with httpx.Client(timeout=25) as client:
            resp = client.post(GROQ_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return {
            "reply": f"I encountered an error: {exc}",
            "recommendations": [],
            "end_of_conversation": False,
        }

    parsed = _extract_json_response(raw_text)

    # Validate recommendations against real catalog
    raw_recs = parsed.get("recommendations", [])
    if raw_recs:
        with open(Path("data/index/entries.pkl"), "rb") as f:
            all_entries = pickle.load(f)
        parsed["recommendations"] = _validate_recommendations(raw_recs, all_entries)
    else:
        parsed["recommendations"] = []

    parsed.setdefault("reply", "")
    parsed.setdefault("end_of_conversation", False)
    return parsed
