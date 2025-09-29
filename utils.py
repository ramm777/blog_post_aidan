import json
from pydantic import BaseModel, Field, ValidationError
from typing import Tuple, Optional
import re

# Pydantic model for reviewer structured output
class ReviewModel(BaseModel):
    Reviewer: str = Field(..., description="Role name of the reviewer (e.g., SEOReviewer)")
    Review: str = Field(..., description="Concise bullet points or feedback text")


def build_summary_args() -> dict:
    """Return summary_args leveraging the schema for strict JSON output."""
    schema = ReviewModel.model_json_schema()
    schema_str = json.dumps(schema["properties"], ensure_ascii=False)
    summary_prompt = (
        "Return ONLY valid JSON matching this schema with required keys Reviewer and Review. "
        f"Schema properties: {schema_str}. No extra keys, no markdown, no explanations."
    )
    return {"summary_prompt": summary_prompt}


def _strip_fences(raw: str) -> str:
    # Remove markdown fences ```json ... ``` or ``` ... ```
    return re.sub(r"```(?:json)?|```", "", raw).strip()


def _attempt_repair(raw: str) -> Optional[str]:
    """Very lightweight repair strategies for common model mistakes."""
    txt = raw.strip()
    if not txt:
        return None
    # Remove leading text before first { and trailing after last }
    if '{' in txt and '}' in txt:
        txt = txt[txt.find('{'): txt.rfind('}')+1]
    # Replace smart quotes
    txt = txt.replace('“', '"').replace('”', '"').replace("'", '"')
    # Remove trailing commas before closing braces/brackets
    txt = re.sub(r",\s*([}\]])", r"\1", txt)
    # If keys missing quotes like Reviewer: add quotes
    txt = re.sub(r"(?m)^(\s*)(Reviewer|Review)\s*:\s*", lambda m: f'{m.group(1)}"{m.group(2)}": ', txt)
    # Ensure minimal JSON shape
    if '"Reviewer"' not in txt and re.search(r"Reviewer\s*:", raw):
        # Already handled above, but fallback
        pass
    return txt


def _fallback_from_lines(raw: str) -> Optional[ReviewModel]:
    """If JSON parsing fails, try to extract lines Reviewer: X / Review: Y."""
    reviewer = None
    review_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"Reviewer\s*[:|-]\s*(.+)", line, re.IGNORECASE)
        if m:
            reviewer = m.group(1).strip()
            continue
        # Collect bullet-ish lines
        if reviewer:
            review_lines.append(line.lstrip('-* '))
    if reviewer and review_lines:
        try:
            return ReviewModel(Reviewer=reviewer[:60], Review='; '.join(review_lines)[:1000])
        except ValidationError:
            return None
    return None


def validate_review(raw: str) -> Tuple[Optional[ReviewModel], Optional[str]]:
    """Attempt to parse & validate a raw JSON string into ReviewModel.
    Returns (model_instance, error_message). Performs:
    1. Fence stripping
    2. Direct JSON parse
    3. Lightweight repair & parse
    4. Fallback heuristic line extraction
    """
    original = raw
    try_order_errors = []
    # Step 1: strip fences
    stripped = _strip_fences(original)
    # Step 2: direct JSON extraction between first { and last }
    try:
        if '{' in stripped and '}' in stripped:
            segment = stripped[stripped.find('{'): stripped.rfind('}')+1]
            model = ReviewModel.model_validate_json(segment)
            return model, None
    except (ValidationError, ValueError) as e:
        try_order_errors.append(f"direct_json: {e}")
    # Step 3: repair
    repaired = _attempt_repair(stripped)
    if repaired:
        try:
            model = ReviewModel.model_validate_json(repaired)
            return model, None
        except (ValidationError, ValueError) as e:
            try_order_errors.append(f"repair: {e}")
    # Step 4: fallback heuristic
    fb = _fallback_from_lines(stripped)
    if fb:
        return fb, None
    # Failure
    return None, "; ".join(try_order_errors) if try_order_errors else "Unparseable reviewer output"


def _to_plain_text(obj):
    """Normalize potential Autogen result objects to plain text."""
    if isinstance(obj, dict) and obj.get("summary"):
        return str(obj["summary"])[:]
    if hasattr(obj, "summary"):
        try:
            val = getattr(obj, "summary")
            if val:
                return str(val)
        except Exception:  # pragma: no cover - defensive
            pass
    return obj if isinstance(obj, str) else str(obj)

# Canonical list of reviewer agent names (excluding meta which is handled separately)
DEFAULT_REVIEWERS = [
    "SEOReviewer",
    "LegalReviewer",
    "EthicsReviewer",
    "FactChecker",
    "BiasReviewer",
    "AccessibilityReviewer",
]
