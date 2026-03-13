"""
extraction.py — Prompt templates for extracting structured data from unstructured text.

Part of PromptKit by Omana Prabhakar (github.com/Omana30)
"""

from typing import Optional
import json


# ─── OPENAI HELPER ───────────────────────────────────────────────────────────

def _call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API with a prompt. Returns response text."""
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        raise ImportError("Run: pip install openai")


# ─── ANTHROPIC HELPER ────────────────────────────────────────────────────────

def _call_anthropic(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Call Anthropic API with a prompt. Returns response text."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except ImportError:
        raise ImportError("Run: pip install anthropic")


# ─── TEMPLATE 1: ENTITY EXTRACTION ───────────────────────────────────────────

def extract_entities(
    text: str,
    entity_types: list[str] = ["person", "organisation", "location", "date"],
    provider: str = "openai"
) -> dict:
    """
    Extract named entities from unstructured text.

    Args:
        text: The input text to extract entities from.
        entity_types: List of entity types to look for.
        provider: "openai" or "anthropic"

    Returns:
        dict: Entity type as key, list of found entities as value.

    Example:
        >>> result = extract_entities(
        ...     text="Dr Sarah Ahmed met with Omana at Heriot-Watt in January 2025.",
        ...     entity_types=["person", "organisation", "date"]
        ... )
        >>> # Returns: {"person": ["Dr Sarah Ahmed", "Omana"], "organisation": ["Heriot-Watt"], "date": ["January 2025"]}
    """
    entity_list = ", ".join(entity_types)

    prompt = f"""Extract all named entities from the text below.

Entity types to find: {entity_list}

Return ONLY a valid JSON object. No explanation. No markdown. No extra text.
Format: {{"entity_type": ["entity1", "entity2"], ...}}
If no entities found for a type, return an empty list.

Text:
{text}

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 2: KEY FACT EXTRACTION ─────────────────────────────────────────

def extract_key_facts(
    text: str,
    facts_to_extract: list[str],
    provider: str = "openai"
) -> dict:
    """
    Extract specific, user-defined facts from text.

    Args:
        text: The input text to extract from.
        facts_to_extract: List of fact labels you want extracted.
        provider: "openai" or "anthropic"

    Returns:
        dict: Fact label as key, extracted value as value. None if not found.

    Example:
        >>> result = extract_key_facts(
        ...     text="Patient reported chest pain for 3 days. No fever. BP 130/85.",
        ...     facts_to_extract=["symptoms", "duration", "blood_pressure", "temperature"]
        ... )
        >>> # Returns: {"symptoms": "chest pain", "duration": "3 days", "blood_pressure": "130/85", "temperature": None}
    """
    facts_formatted = "\n".join([f"- {f}" for f in facts_to_extract])

    prompt = f"""Extract the following specific facts from the text below.

Facts to extract:
{facts_formatted}

Rules:
- Return ONLY a valid JSON object. No markdown. No explanation.
- If a fact is not mentioned in the text, set its value to null.
- Keep extracted values concise and exact — do not paraphrase.
- Format: {{"fact_name": "extracted value or null"}}

Text:
{text}

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 3: ACTION ITEM EXTRACTION ──────────────────────────────────────

def extract_action_items(
    text: str,
    include_owner: bool = True,
    include_deadline: bool = True,
    provider: str = "openai"
) -> list[dict]:
    """
    Extract action items, owners, and deadlines from meeting notes or documents.

    Args:
        text: Meeting notes or document text.
        include_owner: Whether to extract who owns each action.
        include_deadline: Whether to extract deadlines.
        provider: "openai" or "anthropic"

    Returns:
        list[dict]: List of action items with task, owner, deadline fields.

    Example:
        >>> result = extract_action_items(
        ...     text="John will send the report by Friday. Sarah needs to book the venue before next week."
        ... )
        >>> # Returns: [{"task": "Send the report", "owner": "John", "deadline": "Friday"}, ...]
    """
    fields = ["task"]
    if include_owner:
        fields.append("owner")
    if include_deadline:
        fields.append("deadline")

    fields_str = ", ".join([f'"{f}"' for f in fields])

    prompt = f"""Extract all action items from the text below.

Return ONLY a valid JSON array of objects. No markdown. No explanation.
Each object must have these fields: {fields_str}
If owner or deadline is unknown, set value to null.

Text:
{text}

JSON array output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 4: MEDICAL TERM EXTRACTION ─────────────────────────────────────

def extract_medical_terms(
    text: str,
    categories: list[str] = ["symptoms", "diagnoses", "medications", "procedures", "vitals"],
    provider: str = "openai"
) -> dict:
    """
    Extract clinical and medical terminology from healthcare text.

    Args:
        text: Clinical note, transcript, or medical document.
        categories: Medical categories to extract.
        provider: "openai" or "anthropic"

    Returns:
        dict: Category as key, list of found terms as value.

    Example:
        >>> result = extract_medical_terms(
        ...     text="Patient presents with dyspnea and tachycardia. Prescribed metoprolol 50mg. HR 110, BP 145/90."
        ... )
        >>> # Returns: {"symptoms": ["dyspnea", "tachycardia"], "medications": ["metoprolol 50mg"], "vitals": ["HR 110", "BP 145/90"]}
    """
    categories_str = ", ".join(categories)

    prompt = f"""You are a clinical information extraction assistant.
Extract medical terms from the text below, categorised into: {categories_str}

Rules:
- Return ONLY valid JSON. No markdown. No explanation.
- Preserve original clinical terminology exactly as written.
- If no terms found for a category, return empty list.
- Format: {{"category": ["term1", "term2"]}}

Clinical text:
{text}

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 5: SENTIMENT WITH REASONING ────────────────────────────────────

def extract_sentiment_with_reasoning(
    text: str,
    sentiment_scale: str = "positive/negative/neutral",
    provider: str = "openai"
) -> dict:
    """
    Extract sentiment AND the specific reasoning behind it — not just a label.

    Args:
        text: Text to analyse.
        sentiment_scale: Scale to use, e.g. "positive/negative/neutral" or "1-5"
        provider: "openai" or "anthropic"

    Returns:
        dict: {"sentiment": str, "confidence": float, "reasoning": str, "key_phrases": list}

    Example:
        >>> result = extract_sentiment_with_reasoning(
        ...     text="The product mostly works but the onboarding was confusing and support was slow."
        ... )
        >>> # Returns: {"sentiment": "negative", "confidence": 0.7, "reasoning": "...", "key_phrases": [...]}
    """
    prompt = f"""Analyse the sentiment of the text below.

Sentiment scale: {sentiment_scale}

Return ONLY a valid JSON object with exactly these fields:
- "sentiment": the sentiment label from the scale above
- "confidence": a float between 0.0 and 1.0
- "reasoning": one sentence explaining why you assigned this sentiment
- "key_phrases": list of 2-4 phrases from the text that most influenced the sentiment

No markdown. No explanation outside the JSON.

Text:
{text}

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── VALIDATION HELPER ────────────────────────────────────────────────────────

def validate_output(result: dict | list, required_keys: list[str]) -> bool:
    """
    Validate that an extraction result contains all required keys.

    Args:
        result: The dict or list returned from an extraction template.
        required_keys: Keys that must be present.

    Returns:
        bool: True if valid, raises ValueError if not.

    Example:
        >>> validate_output({"sentiment": "positive", "confidence": 0.9}, ["sentiment", "confidence"])
        >>> # Returns: True
    """
    if isinstance(result, list):
        if not result:
            return True
        result = result[0]

    missing = [k for k in required_keys if k not in result]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    return True


# ─── JSON PARSE HELPER ────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict | list:
    """Safely parse JSON from LLM output, stripping markdown fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw output: {raw}")