"""
structured_output.py — Prompt templates for forcing LLMs to return clean, structured output.

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
            temperature=0.1
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
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except ImportError:
        raise ImportError("Run: pip install anthropic")


# ─── TEMPLATE 1: JSON SCHEMA ENFORCEMENT ─────────────────────────────────────

def enforce_json_schema(
    text: str,
    schema: dict,
    strict: bool = True,
    provider: str = "openai"
) -> dict:
    """
    Force LLM output to match a defined JSON schema exactly.

    Args:
        text: Input text to convert into structured JSON.
        schema: The JSON schema dict defining expected output shape.
        strict: If True, raises error on schema mismatch. If False, returns raw.
        provider: "openai" or "anthropic"

    Returns:
        dict: Structured output matching the provided schema.

    Example:
        >>> schema = {
        ...     "name": "string",
        ...     "age": "integer",
        ...     "conditions": ["string"]
        ... }
        >>> result = enforce_json_schema(
        ...     text="John Smith is 45 years old and has diabetes and hypertension.",
        ...     schema=schema
        ... )
        >>> # Returns: {"name": "John Smith", "age": 45, "conditions": ["diabetes", "hypertension"]}
    """
    schema_str = json.dumps(schema, indent=2)

    prompt = f"""Convert the text below into a JSON object that exactly matches this schema.

Required schema:
{schema_str}

Rules:
- Return ONLY valid JSON. No markdown fences. No explanation. No extra fields.
- Match the exact field names from the schema.
- Match the expected data types (string, integer, boolean, array).
- If a value cannot be found in the text, use null.
- Do not invent values not present in the text.

Text:
{text}

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    result = _parse_json(raw)

    if strict:
        _validate_schema(result, schema)

    return result


# ─── TEMPLATE 2: SOAP NOTE GENERATOR ─────────────────────────────────────────

def generate_soap_note(
    transcript: str,
    patient_context: Optional[str] = None,
    provider: str = "openai"
) -> dict:
    """
    Generate a structured clinical SOAP note from a raw consultation transcript.

    SOAP = Subjective, Objective, Assessment, Plan

    Args:
        transcript: Raw text transcript of a clinical consultation.
        patient_context: Optional background e.g. "45F, known hypertensive"
        provider: "openai" or "anthropic"

    Returns:
        dict: SOAP note with keys: subjective, objective, assessment, plan, follow_up

    Example:
        >>> result = generate_soap_note(
        ...     transcript="Patient says she's had a headache for 3 days...",
        ...     patient_context="32F, no known allergies"
        ... )
        >>> # Returns: {"subjective": "...", "objective": "...", "assessment": "...", "plan": "..."}
    """
    context_line = f"\nPatient context: {patient_context}" if patient_context else ""

    prompt = f"""You are a clinical documentation assistant. Generate a structured SOAP note from the consultation transcript below.{context_line}

SOAP format:
- Subjective: What the patient reports — symptoms, history, complaints in their own words
- Objective: Measurable findings — vitals, observations, test results mentioned
- Assessment: Clinical interpretation — likely diagnosis or differential diagnoses
- Plan: Proposed treatment, medications, referrals, follow-up actions
- Follow_up: When and how the patient should be seen next

Rules:
- Return ONLY valid JSON. No markdown. No extra explanation.
- Be clinically precise. Use proper medical terminology.
- If information for a section is not in the transcript, write "Not documented."
- Format:
{{
  "subjective": "...",
  "objective": "...",
  "assessment": "...",
  "plan": "...",
  "follow_up": "..."
}}

Transcript:
{transcript}

SOAP note JSON:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 3: MEETING NOTES FORMATTER ─────────────────────────────────────

def format_meeting_notes(
    raw_notes: str,
    meeting_title: Optional[str] = None,
    attendees: Optional[list[str]] = None,
    provider: str = "openai"
) -> dict:
    """
    Transform raw, unstructured meeting notes into a clean structured format.

    Args:
        raw_notes: Unstructured notes from a meeting.
        meeting_title: Optional title for the meeting.
        attendees: Optional list of attendee names.
        provider: "openai" or "anthropic"

    Returns:
        dict: Structured notes with summary, decisions, actions, next_steps fields.

    Example:
        >>> result = format_meeting_notes(
        ...     raw_notes="We talked about Q4 budget. Sarah said we need to cut 10%. John will send revised forecast...",
        ...     meeting_title="Q4 Budget Review"
        ... )
        >>> # Returns structured dict with summary, decisions, action_items, next_steps
    """
    header = ""
    if meeting_title:
        header += f"Meeting: {meeting_title}\n"
    if attendees:
        header += f"Attendees: {', '.join(attendees)}\n"

    prompt = f"""Structure the following raw meeting notes into a clean, organised format.

{header}
Return ONLY valid JSON with exactly these fields:
{{
  "summary": "2-3 sentence overview of what was discussed",
  "key_decisions": ["decision 1", "decision 2"],
  "action_items": [
    {{"task": "...", "owner": "...", "deadline": "..."}}
  ],
  "next_steps": ["step 1", "step 2"],
  "open_questions": ["question 1", "question 2"]
}}

If a field has no content, return an empty list or "None noted."
No markdown. No explanation outside the JSON.

Raw notes:
{raw_notes}

Structured JSON:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 4: USER STORY GENERATOR ────────────────────────────────────────

def generate_user_story(
    feature_description: str,
    persona: Optional[str] = None,
    include_acceptance_criteria: bool = True,
    provider: str = "openai"
) -> dict:
    """
    Generate properly formatted agile user stories from feature descriptions.

    Args:
        feature_description: Plain English description of a feature or requirement.
        persona: Optional user persona e.g. "therapist", "patient", "admin"
        include_acceptance_criteria: Whether to include acceptance criteria.
        provider: "openai" or "anthropic"

    Returns:
        dict: User story with title, story, acceptance_criteria, priority fields.

    Example:
        >>> result = generate_user_story(
        ...     feature_description="Clinicians need to see their patient notes automatically saved after each session",
        ...     persona="clinician"
        ... )
        >>> # Returns: {"title": "...", "story": "As a clinician, I want...", "acceptance_criteria": [...]}
    """
    persona_line = f"The primary user persona is: {persona}." if persona else ""

    prompt = f"""Generate a properly formatted agile user story from the feature description below.
{persona_line}

Return ONLY valid JSON with these fields:
{{
  "title": "Short feature title (5-8 words)",
  "story": "As a [user], I want [goal] so that [benefit].",
  "acceptance_criteria": [
    "Given [context], when [action], then [outcome].",
    "Given [context], when [action], then [outcome]."
  ],
  "priority": "high/medium/low",
  "story_points": 1-13,
  "notes": "Any technical considerations or edge cases"
}}

{"Include 3-5 acceptance criteria in Given/When/Then format." if include_acceptance_criteria else "Set acceptance_criteria to empty list."}
No markdown. No explanation outside the JSON.

Feature description:
{feature_description}

User story JSON:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 5: BUG REPORT FORMATTER ────────────────────────────────────────

def format_bug_report(
    description: str,
    severity: Optional[str] = None,
    component: Optional[str] = None,
    provider: str = "openai"
) -> dict:
    """
    Transform a vague bug description into a structured, actionable bug report.

    Args:
        description: Vague or unstructured description of the bug.
        severity: Optional severity hint e.g. "crashes app", "minor visual issue"
        component: Optional component e.g. "authentication", "dashboard"
        provider: "openai" or "anthropic"

    Returns:
        dict: Structured bug report with all standard fields.

    Example:
        >>> result = format_bug_report(
        ...     description="The app crashes when I try to upload a PDF on mobile",
        ...     severity="crashes app",
        ...     component="file upload"
        ... )
        >>> # Returns structured bug report with steps to reproduce, expected vs actual, etc.
    """
    hints = ""
    if severity:
        hints += f"\nSeverity hint: {severity}"
    if component:
        hints += f"\nComponent: {component}"

    prompt = f"""Convert the bug description below into a structured, developer-ready bug report.{hints}

Return ONLY valid JSON with exactly these fields:
{{
  "title": "Clear, specific bug title",
  "severity": "critical/high/medium/low",
  "component": "Affected component or area",
  "summary": "One sentence description of the bug",
  "steps_to_reproduce": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "expected_behaviour": "What should happen",
  "actual_behaviour": "What actually happens",
  "possible_cause": "Hypothesis about root cause",
  "suggested_fix": "Suggested resolution if apparent"
}}

If information is missing, make reasonable inferences or write "Unknown - needs investigation."
No markdown. No explanation outside the JSON.

Bug description:
{description}

Bug report JSON:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── VALIDATION HELPERS ───────────────────────────────────────────────────────

def validate_soap_note(note: dict) -> bool:
    """Validate a SOAP note has all required sections."""
    required = ["subjective", "objective", "assessment", "plan", "follow_up"]
    missing = [k for k in required if k not in note]
    if missing:
        raise ValueError(f"SOAP note missing sections: {missing}")
    return True


def _validate_schema(result: dict, schema: dict) -> bool:
    """Validate result contains all top-level keys from schema."""
    missing = [k for k in schema.keys() if k not in result]
    if missing:
        raise ValueError(f"Output missing schema fields: {missing}")
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