"""
summarisation.py — Prompt templates for summarising content in different styles.

Part of PromptKit by Omana Prabhakar (github.com/Omana30)
"""

from typing import Optional
import json


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _call_openai(prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API. Returns response text."""
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        raise ImportError("Run: pip install openai")


def _call_anthropic(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Call Anthropic API. Returns response text."""
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


# ─── TEMPLATE 1: EXECUTIVE SUMMARY ───────────────────────────────────────────

def executive_summary(
    text: str,
    max_sentences: int = 3,
    focus: Optional[str] = None,
    provider: str = "openai"
) -> str:
    """
    Generate a crisp executive summary for decision makers.

    Args:
        text: The content to summarise.
        max_sentences: Maximum number of sentences in the summary.
        focus: Optional focus area e.g. "financial impact" or "risks only"
        provider: "openai" or "anthropic"

    Returns:
        str: A concise executive summary.

    Example:
        >>> result = executive_summary(
        ...     text="Long quarterly report...",
        ...     max_sentences=3,
        ...     focus="key decisions required"
        ... )
        >>> # Returns: "Q3 revenue grew 12% YoY driven by enterprise sales..."
    """
    focus_instruction = f"\nFocus specifically on: {focus}" if focus else ""

    prompt = f"""Write an executive summary of the text below.

Rules:
- Maximum {max_sentences} sentences.
- Write for a senior decision maker who has 30 seconds to read this.
- Lead with the most important finding or outcome.
- Be specific — include numbers and facts where present.
- No filler phrases like "This document discusses..." or "In conclusion..."
- Plain text only. No bullet points. No headers.{focus_instruction}

Text:
{text}

Executive summary:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    return caller(prompt)


# ─── TEMPLATE 2: BULLET SUMMARY ──────────────────────────────────────────────

def bullet_summary(
    text: str,
    max_bullets: int = 5,
    style: str = "concise",
    provider: str = "openai"
) -> list[str]:
    """
    Summarise content as clean bullet points.

    Args:
        text: The content to summarise.
        max_bullets: Maximum number of bullet points to return.
        style: "concise" (one line each) or "detailed" (2-3 lines each)
        provider: "openai" or "anthropic"

    Returns:
        list[str]: List of bullet point strings without bullet characters.

    Example:
        >>> result = bullet_summary(
        ...     text="Meeting transcript...",
        ...     max_bullets=4,
        ...     style="concise"
        ... )
        >>> # Returns: ["Revenue grew 12% in Q3", "New hire planned for engineering", ...]
    """
    style_instruction = (
        "Each bullet must be one clear sentence."
        if style == "concise"
        else "Each bullet can be 2-3 sentences with context."
    )

    prompt = f"""Summarise the text below as bullet points.

Rules:
- Maximum {max_bullets} bullet points.
- {style_instruction}
- Order by importance — most important first.
- Be specific — include numbers and names where present.
- Return ONLY a JSON array of strings. No bullet characters. No markdown. No explanation.
- Format: ["point one", "point two", ...]

Text:
{text}

JSON array:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 3: ONE LINE SUMMARY ────────────────────────────────────────────

def one_line_summary(
    text: str,
    max_words: int = 20,
    provider: str = "openai"
) -> str:
    """
    Distil any content into a single, precise sentence.

    Args:
        text: The content to summarise.
        max_words: Maximum word count for the summary.
        provider: "openai" or "anthropic"

    Returns:
        str: A single sentence summary.

    Example:
        >>> result = one_line_summary(
        ...     text="Long research paper about transformer architectures..."
        ... )
        >>> # Returns: "Transformer models outperform RNNs on NLP tasks by using attention mechanisms."
    """
    prompt = f"""Summarise the text below in exactly ONE sentence of maximum {max_words} words.

Rules:
- Capture the single most important point only.
- Be specific and concrete — avoid vague generalisations.
- No filler phrases. Start directly with the content.
- Plain text only. No punctuation beyond the sentence itself.

Text:
{text}

One sentence summary:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    return caller(prompt).strip().rstrip(".")


# ─── TEMPLATE 4: AUDIENCE SUMMARY ────────────────────────────────────────────

def audience_summary(
    text: str,
    audience: str = "non-technical",
    context: Optional[str] = None,
    provider: str = "openai"
) -> str:
    """
    Summarise content tailored to a specific audience's knowledge level and needs.

    Args:
        text: The content to summarise.
        audience: e.g. "non-technical", "technical", "executive", "clinician", "investor"
        context: Optional context about what the audience needs e.g. "making a budget decision"
        provider: "openai" or "anthropic"

    Returns:
        str: An audience-appropriate summary.

    Example:
        >>> result = audience_summary(
        ...     text="RAG pipeline technical documentation...",
        ...     audience="non-technical",
        ...     context="deciding whether to invest in this system"
        ... )
        >>> # Returns plain English explanation without jargon
    """
    audience_profiles = {
        "non-technical": "Avoid all technical jargon. Use everyday analogies. Focus on outcomes and impact.",
        "technical": "Include technical details, architecture decisions, and implementation specifics.",
        "executive": "Focus on business impact, ROI, risks, and decisions required. No implementation details.",
        "clinician": "Use clinical terminology. Focus on patient outcomes and workflow impact.",
        "investor": "Focus on market opportunity, competitive advantage, traction, and financial potential."
    }

    profile = audience_profiles.get(audience, f"Tailor language and content for a {audience} audience.")
    context_line = f"\nThe audience is specifically: {context}" if context else ""

    prompt = f"""Summarise the text below for a {audience} audience.

Audience guidance: {profile}{context_line}

Rules:
- 3-5 sentences maximum.
- Use language and framing appropriate for this specific audience.
- Focus on what matters most to this audience — not what's technically interesting.
- Plain text. No bullet points. No headers.

Text:
{text}

Summary for {audience} audience:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    return caller(prompt)


# ─── TEMPLATE 5: PROGRESSIVE SUMMARY ─────────────────────────────────────────

def progressive_summary(
    text: str,
    levels: list[str] = ["one_line", "short", "detailed"],
    provider: str = "openai"
) -> dict:
    """
    Generate layered summaries at multiple depths simultaneously.
    Useful when different consumers need different levels of detail.

    Args:
        text: The content to summarise.
        levels: List of summary levels to generate.
                Options: "one_line", "short" (3 sentences), "detailed" (full paragraph)
        provider: "openai" or "anthropic"

    Returns:
        dict: Level name as key, summary text as value.

    Example:
        >>> result = progressive_summary(
        ...     text="Annual report...",
        ...     levels=["one_line", "short", "detailed"]
        ... )
        >>> # Returns: {"one_line": "...", "short": "...", "detailed": "..."}
    """
    level_instructions = {
        "one_line": "one_line: exactly 1 sentence, maximum 20 words",
        "short": "short: exactly 3 sentences, most important points only",
        "detailed": "detailed: full paragraph of 6-8 sentences covering all key points"
    }

    requested = {k: v for k, v in level_instructions.items() if k in levels}
    levels_str = "\n".join([f"- {v}" for v in requested.values()])

    prompt = f"""Generate summaries of the text below at multiple levels of detail.

Levels required:
{levels_str}

Return ONLY a valid JSON object. No markdown. No explanation.
Format: {{"level_name": "summary text"}}

Text:
{text}

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── VALIDATION HELPER ────────────────────────────────────────────────────────

def validate_summary(result: str | list | dict, min_length: int = 10) -> bool:
    """
    Validate that a summary result is non-empty and meets minimum length.

    Args:
        result: The summary output to validate.
        min_length: Minimum character length required.

    Returns:
        bool: True if valid, raises ValueError if not.
    """
    if isinstance(result, str):
        if len(result.strip()) < min_length:
            raise ValueError(f"Summary too short: {len(result)} chars (minimum {min_length})")
    elif isinstance(result, list):
        if not result:
            raise ValueError("Summary list is empty")
    elif isinstance(result, dict):
        if not result:
            raise ValueError("Summary dict is empty")
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