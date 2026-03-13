"""
chain_of_thought.py — Prompt templates for structured reasoning and step-by-step thinking.

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
            temperature=0.2
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


# ─── TEMPLATE 1: PROBLEM DECOMPOSITION ───────────────────────────────────────

def decompose_problem(
    problem: str,
    max_steps: int = 5,
    domain: Optional[str] = None,
    provider: str = "openai"
) -> dict:
    """
    Break a complex problem into clear, actionable sub-steps.

    Args:
        problem: The complex problem or challenge to decompose.
        max_steps: Maximum number of sub-steps to generate.
        domain: Optional domain context e.g. "software engineering", "clinical"
        provider: "openai" or "anthropic"

    Returns:
        dict: {"problem_summary": str, "steps": list, "dependencies": list, "risks": list}

    Example:
        >>> result = decompose_problem(
        ...     problem="Build a real-time transcription system for clinical consultations",
        ...     domain="software engineering"
        ... )
        >>> # Returns structured breakdown with steps, dependencies and risks
    """
    domain_line = f"\nDomain context: {domain}" if domain else ""

    prompt = f"""Break down the following problem into clear, actionable steps using structured reasoning.{domain_line}

Problem:
{problem}

Think through this carefully. Then return ONLY valid JSON with these fields:
{{
  "problem_summary": "One sentence restatement of the core problem",
  "steps": [
    {{
      "step_number": 1,
      "title": "Short step title",
      "description": "What to do and why",
      "estimated_effort": "low/medium/high"
    }}
  ],
  "dependencies": ["Any external dependencies or prerequisites"],
  "risks": ["Key risks or blockers to watch for"],
  "success_criteria": "How you know the problem is solved"
}}

Maximum {max_steps} steps. Order by logical sequence.
No markdown. No explanation outside the JSON.

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 2: PROS AND CONS ANALYSIS ──────────────────────────────────────

def pros_cons_analysis(
    decision: str,
    context: Optional[str] = None,
    stakeholders: Optional[list[str]] = None,
    provider: str = "openai"
) -> dict:
    """
    Generate a balanced, structured pros and cons analysis for any decision.

    Args:
        decision: The decision or option being evaluated.
        context: Optional background context for the decision.
        stakeholders: Optional list of stakeholders to consider perspectives of.
        provider: "openai" or "anthropic"

    Returns:
        dict: {"decision": str, "pros": list, "cons": list, "verdict": str, "confidence": float}

    Example:
        >>> result = pros_cons_analysis(
        ...     decision="Switch our ML pipeline from PyTorch to JAX",
        ...     context="Small team, production system, 6 month timeline",
        ...     stakeholders=["engineering team", "product manager"]
        ... )
        >>> # Returns balanced analysis with verdict and confidence score
    """
    context_line = f"\nContext: {context}" if context else ""
    stakeholder_line = f"\nConsider perspectives of: {', '.join(stakeholders)}" if stakeholders else ""

    prompt = f"""Perform a balanced pros and cons analysis for the decision below.{context_line}{stakeholder_line}

Decision:
{decision}

Think through both sides carefully and honestly. Do not favour one side.

Return ONLY valid JSON:
{{
  "decision": "Restated decision clearly",
  "pros": [
    {{"point": "Pro point", "impact": "high/medium/low", "reasoning": "Why this matters"}}
  ],
  "cons": [
    {{"point": "Con point", "impact": "high/medium/low", "reasoning": "Why this matters"}}
  ],
  "key_trade_off": "The single most important trade-off to understand",
  "verdict": "Recommended course of action in one sentence",
  "confidence": 0.0
}}

Set confidence between 0.0 (very uncertain) and 1.0 (very certain).
Include 3-5 pros and 3-5 cons minimum.
No markdown. No explanation outside the JSON.

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 3: DECISION FRAMEWORK ──────────────────────────────────────────

def decision_framework(
    question: str,
    options: list[str],
    criteria: Optional[list[str]] = None,
    provider: str = "openai"
) -> dict:
    """
    Score multiple options against defined criteria to support structured decision making.

    Args:
        question: The decision question e.g. "Which database should we use?"
        options: List of options to evaluate.
        criteria: Optional evaluation criteria. If None, LLM generates relevant criteria.
        provider: "openai" or "anthropic"

    Returns:
        dict: Scored comparison matrix with recommendation.

    Example:
        >>> result = decision_framework(
        ...     question="Which vector database should we use for our RAG pipeline?",
        ...     options=["Pinecone", "ChromaDB", "Weaviate"],
        ...     criteria=["cost", "ease of setup", "scalability", "Python support"]
        ... )
        >>> # Returns scored matrix and recommendation
    """
    options_str = "\n".join([f"- {o}" for o in options])
    criteria_line = (
        f"\nEvaluation criteria to use:\n" + "\n".join([f"- {c}" for c in criteria])
        if criteria
        else "\nGenerate the most relevant evaluation criteria for this decision."
    )

    prompt = f"""Evaluate the following options to answer the decision question below.

Decision question: {question}

Options to evaluate:
{options_str}
{criteria_line}

Score each option against each criterion from 1 (poor) to 5 (excellent).
Provide brief reasoning for each score.

Return ONLY valid JSON:
{{
  "question": "The decision question",
  "criteria": ["criterion1", "criterion2"],
  "evaluations": [
    {{
      "option": "Option name",
      "scores": {{"criterion1": 4, "criterion2": 3}},
      "total_score": 7,
      "summary": "One sentence summary of this option"
    }}
  ],
  "recommendation": "The recommended option",
  "reasoning": "Why this option wins overall"
}}

No markdown. No explanation outside the JSON.

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 4: ROOT CAUSE ANALYSIS ─────────────────────────────────────────

def root_cause_analysis(
    problem: str,
    context: Optional[str] = None,
    depth: int = 5,
    provider: str = "openai"
) -> dict:
    """
    Perform a Five Whys style root cause analysis on any problem.

    Args:
        problem: The observable problem or symptom to investigate.
        context: Optional context about the system or situation.
        depth: Number of "why" levels to drill down (default 5).
        provider: "openai" or "anthropic"

    Returns:
        dict: {"problem": str, "why_chain": list, "root_cause": str, "recommendations": list}

    Example:
        >>> result = root_cause_analysis(
        ...     problem="Our RAG pipeline is returning irrelevant answers",
        ...     context="Using ChromaDB with OpenAI embeddings, 10k documents"
        ... )
        >>> # Returns five-whys chain down to root cause with recommendations
    """
    context_line = f"\nContext: {context}" if context else ""

    prompt = f"""Perform a root cause analysis using the Five Whys method on the problem below.{context_line}

Problem:
{problem}

Drill down {depth} levels deep, asking "why" at each step.
Be specific and analytical — do not give generic answers.

Return ONLY valid JSON:
{{
  "problem_statement": "Clear restatement of the problem",
  "why_chain": [
    {{
      "level": 1,
      "question": "Why does [problem] occur?",
      "answer": "Because..."
    }}
  ],
  "root_cause": "The fundamental root cause identified",
  "contributing_factors": ["Factor 1", "Factor 2"],
  "recommendations": [
    {{
      "action": "Specific action to take",
      "addresses": "Which part of the root cause this fixes",
      "priority": "immediate/short-term/long-term"
    }}
  ]
}}

No markdown. No explanation outside the JSON.

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 5: STEP BY STEP WRAPPER ────────────────────────────────────────

def step_by_step_wrapper(
    task: str,
    show_reasoning: bool = True,
    output_format: str = "text",
    provider: str = "openai"
) -> dict:
    """
    Wrap any task or question in a chain-of-thought reasoning prompt.
    Forces the model to think before answering — significantly improves accuracy
    on complex tasks.

    Args:
        task: Any question, task, or problem to solve with CoT reasoning.
        show_reasoning: If True, returns the reasoning chain alongside the answer.
        output_format: "text" for plain answer, "json" for structured output.
        provider: "openai" or "anthropic"

    Returns:
        dict: {"reasoning_steps": list, "answer": str, "confidence": float}

    Example:
        >>> result = step_by_step_wrapper(
        ...     task="Should I use a vector database or a traditional SQL database for storing embeddings?",
        ...     show_reasoning=True
        ... )
        >>> # Returns step-by-step reasoning chain and final answer
    """
    reasoning_instruction = (
        "Show your full reasoning process step by step before giving the final answer."
        if show_reasoning
        else "Think through this internally, then give only the final answer."
    )

    prompt = f"""Answer the following by thinking carefully step by step.

{reasoning_instruction}

Task:
{task}

Return ONLY valid JSON:
{{
  "reasoning_steps": [
    {{
      "step": 1,
      "thought": "What you're considering at this step"
    }}
  ],
  "answer": "Your final, definitive answer",
  "confidence": 0.0,
  "caveats": ["Any important caveats or conditions on this answer"]
}}

Set confidence between 0.0 (very uncertain) and 1.0 (very certain).
No markdown. No explanation outside the JSON.

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── VALIDATION HELPER ────────────────────────────────────────────────────────

def validate_reasoning_output(result: dict, required_keys: list[str]) -> bool:
    """
    Validate that a chain of thought result has all required keys.

    Args:
        result: Output dict from any CoT template.
        required_keys: Keys that must be present.

    Returns:
        bool: True if valid, raises ValueError if not.
    """
    missing = [k for k in required_keys if k not in result]
    if missing:
        raise ValueError(f"Reasoning output missing keys: {missing}")
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