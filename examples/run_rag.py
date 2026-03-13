"""
run_rag.py — Runnable examples for all RAG query templates.

Demonstrates advanced RAG techniques including HyDE, query rewriting,
citation grounding, fallback handling, and multi-hop reasoning.

Run this file directly:
    python examples/run_rag.py

Requires OPENAI_API_KEY in your .env file or environment variables.

Part of PromptKit by Omana Prabhakar (github.com/Omana30)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from templates.rag_queries import (
    rewrite_query,
    hypothetical_document,
    answer_with_citations,
    no_answer_fallback,
    multi_hop_reasoning
)

SAMPLE_CHUNKS = [
    {
        "id": "chunk_001",
        "source": "NHS_Clinical_Guidelines_2024.pdf",
        "text": """Metformin is the first-line pharmacological treatment for type 2 diabetes
        in adults. It works by reducing hepatic glucose production and improving
        insulin sensitivity. Standard starting dose is 500mg twice daily with meals,
        titrated up to 2000mg daily based on tolerability and glycaemic response."""
    },
    {
        "id": "chunk_002",
        "source": "BNF_Elderly_Prescribing.pdf",
        "text": """In patients over 65, renal function must be assessed before initiating
        metformin. The drug is contraindicated when eGFR falls below 30 mL/min/1.73m².
        Dose reduction is recommended when eGFR is between 30-45 mL/min/1.73m².
        Regular monitoring of renal function every 3-6 months is advised."""
    },
    {
        "id": "chunk_003",
        "source": "Diabetes_UK_Patient_Guidelines.pdf",
        "text": """Common side effects of metformin include nausea, diarrhoea, and abdominal
        discomfort, particularly when starting treatment. These gastrointestinal effects
        are typically transient and can be minimised by taking the medication with food
        and titrating the dose slowly. Lactic acidosis is a rare but serious complication."""
    }
]


def example_query_rewriting():
    print("\n" + "="*60)
    print("EXAMPLE 1: Query Rewriting for Better Retrieval")
    print("="*60)

    result = rewrite_query(
        original_query="is metformin safe for old people?",
        context="clinical guidelines RAG system for NHS prescribers",
        num_variants=3
    )

    print(f"Original query: {result['original']}")
    print(f"\nRewritten variants:")
    for i, variant in enumerate(result['rewritten_queries'], 1):
        print(f"\n  Variant {i}: {variant['query']}")
        print(f"  Rationale:  {variant['rationale']}")
    print(f"\nBest query for retrieval: {result['best_query']}")


def example_hyde():
    print("\n" + "="*60)
    print("EXAMPLE 2: HyDE — Hypothetical Document Embedding")
    print("="*60)
    print("(Generate a hypothetical ideal document to embed instead of the raw query)")

    result = hypothetical_document(
        query="What are the renal considerations for metformin in elderly patients?",
        document_type="clinical guideline",
        length="paragraph"
    )

    print(f"\nQuery: {result['query']}")
    print(f"\nHypothetical document (embed this for retrieval):")
    print(f"  {result['hypothetical_document']}")
    print(f"\nKey retrieval terms: {result['key_terms']}")


def example_answer_with_citations():
    print("\n" + "="*60)
    print("EXAMPLE 3: Answer with Citations (Grounded RAG)")
    print("="*60)

    result = answer_with_citations(
        query="What should I know before prescribing metformin to a 72-year-old patient?",
        retrieved_chunks=SAMPLE_CHUNKS,
        strict_grounding=True
    )

    print(f"Query: What should I know before prescribing metformin to a 72-year-old?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nCitations:")
    for citation in result.get('citations', []):
        print(f"  [{citation['chunk_id']}] {citation['claim']}")
        print(f"  Quote: \"{citation['quote']}\"")
    print(f"\nConfidence: {result.get('confidence')}")
    if result.get('gaps'):
        print(f"Information gaps: {result['gaps']}")


def example_no_answer_fallback():
    print("\n" + "="*60)
    print("EXAMPLE 4: No-Answer Fallback (Graceful Handling)")
    print("="*60)

    irrelevant_chunks = [
        {
            "id": "chunk_x",
            "source": "Cardiology_Guidelines.pdf",
            "text": "Beta blockers are first-line treatment for stable angina. Bisoprolol 5mg once daily is recommended as initial therapy."
        }
    ]

    result = no_answer_fallback(
        query="What is the patient's current HbA1c level?",
        retrieved_chunks=irrelevant_chunks
    )

    print(f"Query: What is the patient's current HbA1c level?")
    print(f"\nCan answer: {result.get('can_answer')}")
    print(f"Relevance score: {result.get('relevance_score')}")
    print(f"Response: {result.get('answer')}")
    print(f"What's missing: {result.get('what_is_missing')}")
    print(f"Suggestions: {result.get('suggestions')}")


def example_multi_hop_reasoning():
    print("\n" + "="*60)
    print("EXAMPLE 5: Multi-Hop Reasoning Across Chunks")
    print("="*60)

    result = multi_hop_reasoning(
        query="Given that a patient is 70 years old with reduced kidney function, what is the safest metformin approach?",
        chunks=SAMPLE_CHUNKS,
        max_hops=3
    )

    print(f"Query: {result['query']}")
    print(f"\nReasoning chain:")
    for hop in result.get('reasoning_chain', []):
        print(f"\n  Hop {hop['hop']} [{hop['source_used']}]:")
        print(f"  Found:     {hop['information_extracted']}")
        print(f"  Connects:  {hop['connection']}")
    print(f"\nFinal answer: {result['answer']}")
    print(f"Confidence: {result.get('confidence')}")
    if result.get('assumptions_made'):
        print(f"Assumptions: {result['assumptions_made']}")


if __name__ == "__main__":
    print("PromptKit — RAG Query Templates Demo")
    print("Demonstrating advanced RAG techniques...\n")

    try:
        example_query_rewriting()
        example_hyde()
        example_answer_with_citations()
        example_no_answer_fallback()
        example_multi_hop_reasoning()
        print("\n" + "="*60)
        print("All RAG examples completed successfully.")
        print("="*60)
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your OPENAI_API_KEY is set in .env")