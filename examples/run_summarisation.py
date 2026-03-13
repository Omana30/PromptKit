"""
run_summarisation.py — Runnable examples for all summarisation templates.

Run this file directly:
    python examples/run_summarisation.py

Requires OPENAI_API_KEY in your .env file or environment variables.

Part of PromptKit by Omana Prabhakar (github.com/Omana30)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from templates.summarisation import (
    executive_summary,
    bullet_summary,
    one_line_summary,
    audience_summary,
    progressive_summary
)

SAMPLE_TEXT = """
The UK government has announced a £1.5 billion investment in artificial intelligence
infrastructure over the next three years. The funding will be directed toward
three priority areas: compute infrastructure at national AI research labs,
clinical AI deployment across NHS trusts, and an AI skills programme targeting
100,000 workers in industries most affected by automation.

The announcement follows the publication of the National AI Strategy review,
which found that the UK risks falling behind the United States and China in
foundational AI capabilities unless significant public investment is made.
Industry leaders have broadly welcomed the investment, though some have
called for clearer regulatory frameworks before large-scale NHS deployment.

The clinical AI programme is expected to focus initially on radiology,
pathology, and administrative automation — areas where AI has demonstrated
the strongest evidence base. Early pilots across three NHS trusts showed
a 34% reduction in diagnostic reporting times and a 22% reduction in
administrative workload for clinical staff.
"""


def example_executive_summary():
    print("\n" + "="*60)
    print("EXAMPLE 1: Executive Summary")
    print("="*60)
    result = executive_summary(
        text=SAMPLE_TEXT,
        max_sentences=3,
        focus="investment impact and risks"
    )
    print(f"Result:\n{result}")


def example_bullet_summary():
    print("\n" + "="*60)
    print("EXAMPLE 2: Bullet Summary")
    print("="*60)
    result = bullet_summary(
        text=SAMPLE_TEXT,
        max_bullets=4,
        style="concise"
    )
    print("Result:")
    for i, point in enumerate(result, 1):
        print(f"  {i}. {point}")


def example_one_line_summary():
    print("\n" + "="*60)
    print("EXAMPLE 3: One Line Summary")
    print("="*60)
    result = one_line_summary(text=SAMPLE_TEXT, max_words=20)
    print(f"Result: {result}")


def example_audience_summary():
    print("\n" + "="*60)
    print("EXAMPLE 4: Audience Summary (Non-Technical vs Technical)")
    print("="*60)

    non_tech = audience_summary(
        text=SAMPLE_TEXT,
        audience="non-technical",
        context="deciding whether to support this policy"
    )
    print(f"Non-technical audience:\n{non_tech}")

    print()

    technical = audience_summary(
        text=SAMPLE_TEXT,
        audience="technical",
        context="scoping an NHS AI implementation project"
    )
    print(f"Technical audience:\n{technical}")


def example_progressive_summary():
    print("\n" + "="*60)
    print("EXAMPLE 5: Progressive Summary (Multiple Depths)")
    print("="*60)
    result = progressive_summary(
        text=SAMPLE_TEXT,
        levels=["one_line", "short", "detailed"]
    )
    for level, summary in result.items():
        print(f"\n  [{level.upper()}]")
        print(f"  {summary}")


if __name__ == "__main__":
    print("PromptKit — Summarisation Templates Demo")
    print("Running all summarisation examples...\n")

    try:
        example_executive_summary()
        example_bullet_summary()
        example_one_line_summary()
        example_audience_summary()
        example_progressive_summary()
        print("\n" + "="*60)
        print("All examples completed successfully.")
        print("="*60)
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your OPENAI_API_KEY is set in .env")