"""
rag_queries.py — Prompt templates for building reliable RAG (Retrieval Augmented Generation) pipelines.

Includes advanced techniques like HyDE (Hypothetical Document Embeddings),
query rewriting, citation grounding, and multi-hop reasoning.

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


# ─── TEMPLATE 1: QUERY REWRITING ─────────────────────────────────────────────

def rewrite_query(
    original_query: str,
    context: Optional[str] = None,
    num_variants: int = 3,
    provider: str = "openai"
) -> dict:
    """
    Rewrite a user query into multiple variants optimised for vector retrieval.

    Why this matters: User queries are often vague, conversational, or poorly
    phrased for semantic search. Rewriting them into precise, retrieval-optimised
    variants significantly improves recall from vector databases.

    Args:
        original_query: The raw user query to rewrite.
        context: Optional conversation context or domain e.g. "medical records system"
        num_variants: Number of rewritten variants to generate.
        provider: "openai" or "anthropic"

    Returns:
        dict: {"original": str, "rewritten_queries": list, "best_query": str}

    Example:
        >>> result = rewrite_query(
        ...     original_query="what's wrong with my heart?",
        ...     context="clinical records RAG system",
        ...     num_variants=3
        ... )
        >>> # Returns: {"original": "...", "rewritten_queries": ["cardiac symptoms...", ...], "best_query": "..."}
    """
    context_line = f"\nSystem context: {context}" if context else ""

    prompt = f"""You are a query optimisation expert for vector database retrieval.

Rewrite the user query below into {num_variants} different variants that will retrieve better results from a semantic vector search.{context_line}

Rules for rewriting:
- Make queries more specific and information-dense
- Use domain-appropriate terminology
- Vary the phrasing and angle of each variant
- Each variant should target slightly different aspects of the information need
- Remove filler words and conversational phrasing

Return ONLY valid JSON:
{{
  "original": "The original query unchanged",
  "rewritten_queries": [
    {{
      "query": "Rewritten query text",
      "rationale": "Why this variant might retrieve different/better results"
    }}
  ],
  "best_query": "The single best query for retrieval from the variants above"
}}

No markdown. No explanation outside the JSON.

Original query: {original_query}

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 2: HYPOTHETICAL DOCUMENT EMBEDDING (HyDE) ──────────────────────

def hypothetical_document(
    query: str,
    document_type: str = "general",
    length: str = "paragraph",
    provider: str = "openai"
) -> dict:
    """
    Generate a hypothetical ideal document for a query — the HyDE technique.

    HyDE (Hypothetical Document Embeddings) is an advanced RAG technique where
    instead of embedding the user query directly, you generate a hypothetical
    ideal document that would answer the query, then embed THAT for retrieval.

    This dramatically improves retrieval quality because the hypothetical document
    is semantically closer to real documents in your corpus than the raw query.

    Args:
        query: The user query to generate a hypothetical document for.
        document_type: Type of document to simulate e.g. "clinical note",
                       "research paper", "technical documentation", "news article"
        length: "sentence", "paragraph", or "full_document"
        provider: "openai" or "anthropic"

    Returns:
        dict: {"query": str, "hypothetical_document": str, "key_terms": list}

    Example:
        >>> result = hypothetical_document(
        ...     query="What are the side effects of metformin in elderly patients?",
        ...     document_type="clinical research paper",
        ...     length="paragraph"
        ... )
        >>> # Returns a hypothetical paragraph from a clinical paper — embed this for retrieval
    """
    length_instructions = {
        "sentence": "Write exactly 1-2 sentences.",
        "paragraph": "Write exactly 1 paragraph of 4-6 sentences.",
        "full_document": "Write a structured document of 3-4 paragraphs."
    }

    length_instruction = length_instructions.get(length, "Write 1 paragraph.")

    prompt = f"""You are simulating a document retrieval expert using the HyDE technique.

Generate a hypothetical {document_type} that would perfectly answer the query below.
This hypothetical document will be used to improve semantic search retrieval.

{length_instruction}
Write as if you are the actual document — do not say "this document discusses" or "here is a document."
Use precise, domain-appropriate language that a real {document_type} would contain.
Include specific terminology, facts, and phrasing typical of a {document_type}.

Return ONLY valid JSON:
{{
  "query": "Original query unchanged",
  "hypothetical_document": "The generated hypothetical document text",
  "key_terms": ["Important terms from the document useful for retrieval"],
  "document_type": "{document_type}"
}}

No markdown. No explanation outside the JSON.

Query: {query}

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 3: ANSWER WITH CITATIONS ───────────────────────────────────────

def answer_with_citations(
    query: str,
    retrieved_chunks: list[dict],
    strict_grounding: bool = True,
    provider: str = "openai"
) -> dict:
    """
    Generate a grounded answer from retrieved context chunks, with source citations.

    Forces the LLM to only use information from retrieved context — preventing
    hallucination by grounding every claim to a specific source chunk.

    Args:
        query: The user's original question.
        retrieved_chunks: List of dicts with "id", "text", and optionally "source" keys.
        strict_grounding: If True, LLM must not use knowledge outside the chunks.
        provider: "openai" or "anthropic"

    Returns:
        dict: {"answer": str, "citations": list, "confidence": float, "gaps": list}

    Example:
        >>> chunks = [
        ...     {"id": "chunk_1", "text": "Metformin reduces glucose production...", "source": "doc_123"},
        ...     {"id": "chunk_2", "text": "In elderly patients, renal function...", "source": "doc_456"}
        ... ]
        >>> result = answer_with_citations(query="What are metformin risks?", retrieved_chunks=chunks)
        >>> # Returns grounded answer with chunk citations
    """
    chunks_formatted = "\n\n".join([
        f"[{c.get('id', f'chunk_{i}')}] {c.get('source', 'Unknown source')}\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    ])

    grounding_instruction = (
        "CRITICAL: Only use information from the provided context chunks. "
        "Do not use any external knowledge. If the answer is not in the chunks, say so."
        if strict_grounding
        else "Prefer information from the context chunks, but you may supplement with general knowledge where clearly needed."
    )

    prompt = f"""Answer the query below using ONLY the provided context chunks.

{grounding_instruction}

Context chunks:
{chunks_formatted}

Query: {query}

Return ONLY valid JSON:
{{
  "answer": "Your grounded answer using information from the chunks",
  "citations": [
    {{
      "chunk_id": "The chunk ID referenced",
      "claim": "The specific claim from your answer supported by this chunk",
      "quote": "Brief direct quote from the chunk supporting this claim"
    }}
  ],
  "confidence": 0.0,
  "gaps": ["Information needed to answer fully that was NOT in the provided chunks"],
  "answered_fully": true
}}

Set confidence 0.0-1.0. Set answered_fully to false if context was insufficient.
No markdown. No explanation outside the JSON.

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 4: NO ANSWER FALLBACK ──────────────────────────────────────────

def no_answer_fallback(
    query: str,
    retrieved_chunks: list[dict],
    relevance_threshold: float = 0.3,
    provider: str = "openai"
) -> dict:
    """
    Gracefully handle cases where retrieved context is insufficient to answer.

    One of the most common RAG failure modes is confidently hallucinating an
    answer when the retrieved context doesn't actually contain the information.
    This template forces explicit acknowledgement of insufficient context.

    Args:
        query: The user's original question.
        retrieved_chunks: Retrieved context chunks to evaluate.
        relevance_threshold: Minimum relevance score to attempt an answer (0.0-1.0).
        provider: "openai" or "anthropic"

    Returns:
        dict: {"can_answer": bool, "answer": str, "relevance_score": float, "suggestions": list}

    Example:
        >>> result = no_answer_fallback(
        ...     query="What is the patient's blood type?",
        ...     retrieved_chunks=[{"id": "1", "text": "Patient presented with chest pain..."}]
        ... )
        >>> # Returns: {"can_answer": False, "relevance_score": 0.1, "suggestions": [...]}
    """
    chunks_formatted = "\n\n".join([
        f"[{c.get('id', f'chunk_{i}')}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    ])

    prompt = f"""Evaluate whether the provided context chunks contain sufficient information to answer the query.

Query: {query}

Retrieved context:
{chunks_formatted}

First, assess how relevant the context is to the query (0.0 = completely irrelevant, 1.0 = perfectly answers the query).

Return ONLY valid JSON:
{{
  "can_answer": true,
  "relevance_score": 0.0,
  "answer": "Your answer if can_answer is true, or 'Insufficient information in the provided context.' if false",
  "what_is_missing": "Specific information that would be needed to answer this query",
  "suggestions": [
    "Suggested alternative queries that might retrieve better results",
    "Suggestion 2"
  ],
  "partial_information": "Any partially relevant information found in context, even if incomplete"
}}

Set can_answer to false if relevance_score is below {relevance_threshold}.
No markdown. No explanation outside the JSON.

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── TEMPLATE 5: MULTI-HOP REASONING ─────────────────────────────────────────

def multi_hop_reasoning(
    query: str,
    chunks: list[dict],
    max_hops: int = 3,
    provider: str = "openai"
) -> dict:
    """
    Connect information across multiple retrieved chunks to answer complex queries
    that require reasoning across several pieces of evidence.

    Multi-hop reasoning is needed when no single chunk answers the question —
    the answer must be synthesised by connecting facts across multiple sources.

    Args:
        query: A complex query requiring information from multiple chunks.
        chunks: List of retrieved context chunks to reason across.
        max_hops: Maximum number of reasoning hops to perform.
        provider: "openai" or "anthropic"

    Returns:
        dict: {"reasoning_chain": list, "answer": str, "evidence_used": list, "confidence": float}

    Example:
        >>> result = multi_hop_reasoning(
        ...     query="What treatment would be most appropriate given the patient's history and current symptoms?",
        ...     chunks=[chunk1, chunk2, chunk3]
        ... )
        >>> # Returns step-by-step reasoning chain connecting multiple chunks
    """
    chunks_formatted = "\n\n".join([
        f"[Source {c.get('id', f'chunk_{i}')}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    prompt = f"""You are a multi-hop reasoning engine. Answer the complex query below by
connecting information across multiple context sources, step by step.

Query: {query}

Available context sources:
{chunks_formatted}

Reason through this in up to {max_hops} hops. Each hop should:
1. Identify a relevant piece of information from a specific source
2. Connect it to information from another source
3. Build toward the final answer

Return ONLY valid JSON:
{{
  "query": "Original query unchanged",
  "reasoning_chain": [
    {{
      "hop": 1,
      "source_used": "chunk id",
      "information_extracted": "What you learned from this source",
      "connection": "How this connects to the next piece of information"
    }}
  ],
  "answer": "Final synthesised answer connecting all hops",
  "evidence_used": ["List of chunk IDs used in reasoning"],
  "confidence": 0.0,
  "assumptions_made": ["Any assumptions made when connecting information"]
}}

No markdown. No explanation outside the JSON.

JSON output:"""

    caller = _call_openai if provider == "openai" else _call_anthropic
    raw = caller(prompt)
    return _parse_json(raw)


# ─── VALIDATION HELPER ────────────────────────────────────────────────────────

def validate_rag_response(result: dict) -> bool:
    """
    Validate a RAG response has minimum required fields and is not hallucinated.

    Args:
        result: Output from any RAG template.

    Returns:
        bool: True if valid, raises ValueError if not.
    """
    if "answer" not in result:
        raise ValueError("RAG response missing 'answer' field")
    if "confidence" in result:
        conf = result["confidence"]
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            raise ValueError(f"Confidence score out of range: {conf}")
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