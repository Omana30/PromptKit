# PromptKit 🧠

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> A library of reusable, battle-tested prompt templates for building reliable LLM-powered applications.

---

## Why this exists

While building clinical AI tools and financial NLP pipelines, I kept rewriting the same prompt patterns from scratch — extraction chains, structured output formatters, RAG query templates. Every project. Every time.

PromptKit is my answer to that. A single, well-documented library of prompt templates I've tested across real projects — so I (and anyone else) can stop reinventing the wheel and start building faster.

These aren't theoretical prompts. Every template here has been used in production or research contexts, refined through iteration, and documented with real examples.

---

## Quick start

```bash
pip install openai anthropic python-dotenv
git clone https://github.com/Omana30/PromptKit.git
cd PromptKit
```

```python
from templates.extraction import extract_key_facts

result = extract_key_facts(
    text="Patient reported chest pain for 3 days. No fever. BP 130/85.",
    facts_to_extract=["symptoms", "duration", "vitals"]
)
print(result)
```

---

## Template catalogue

### 🔍 Extraction (`templates/extraction.py`)
| Template | What it does |
|---|---|
| `extract_entities()` | Pulls named entities from unstructured text |
| `extract_key_facts()` | Extracts specific facts you define |
| `extract_action_items()` | Finds tasks and owners from meeting notes |
| `extract_medical_terms()` | Clinical terminology extraction |
| `extract_sentiment_with_reasoning()` | Sentiment + why, not just a label |

### 📝 Summarisation (`templates/summarisation.py`)
| Template | What it does |
|---|---|
| `executive_summary()` | Crisp 3-sentence summary for decision makers |
| `bullet_summary()` | Clean bullet points from long content |
| `one_line_summary()` | Single sentence distillation |
| `audience_summary()` | Adjusts tone for technical vs non-technical |
| `progressive_summary()` | Layered summaries at different depths |

### 📦 Structured Output (`templates/structured_output.py`)
| Template | What it does |
|---|---|
| `enforce_json_schema()` | Forces LLM output into a defined JSON shape |
| `generate_soap_note()` | Clinical SOAP note from raw transcript |
| `format_meeting_notes()` | Structures raw notes into agenda/actions |
| `generate_user_story()` | Agile user stories from feature descriptions |
| `format_bug_report()` | Structured bug reports from vague descriptions |

### 🧩 Chain of Thought (`templates/chain_of_thought.py`)
| Template | What it does |
|---|---|
| `decompose_problem()` | Breaks complex problems into steps |
| `pros_cons_analysis()` | Balanced analysis of any decision |
| `decision_framework()` | Structured decision-making prompt |
| `root_cause_analysis()` | Five-whys style root cause finder |
| `step_by_step_wrapper()` | Wraps any prompt in CoT reasoning |

### 🔗 RAG Queries (`templates/rag_queries.py`)
| Template | What it does |
|---|---|
| `rewrite_query()` | Rewrites user query for better retrieval |
| `hypothetical_document()` | HyDE technique for dense retrieval |
| `answer_with_citations()` | Forces grounded answers with source refs |
| `no_answer_fallback()` | Graceful handling when context is insufficient |
| `multi_hop_reasoning()` | Connects information across multiple chunks |

---

## Prompt design philosophy

Three rules every template in this library follows:

**1. Explicit over implicit.** Every template tells the model exactly what format to return. No hoping it figures it out.

**2. Validate, don't trust.** Where output format matters, templates include a `validate_output()` helper that checks structure before your code uses it.

**3. Model-agnostic.** Every template works with OpenAI and Anthropic APIs. The prompt logic doesn't change — only the client call.

---

## Project structure

```
PromptKit/
├── templates/          # Core prompt template library
│   ├── extraction.py
│   ├── summarisation.py
│   ├── structured_output.py
│   ├── chain_of_thought.py
│   └── rag_queries.py
├── examples/           # Runnable examples for each template category
│   ├── run_extraction.py
│   ├── run_summarisation.py
│   └── run_rag.py
├── tests/              # Unit tests for template structure validation
│   └── test_templates.py
├── docs/
│   └── TEMPLATES.md    # Full documentation with input/output examples
├── requirements.txt
└── README.md
```

---

## Contributing

Found a pattern that works really well? Open a PR. Every template should include:
- A clear docstring
- Type hints
- At least one usage example
- Works with both OpenAI and Anthropic

---

## License

MIT — use freely, build something great.

---

*Built by [Omana Prabhakar](https://github.com/Omana30) — AI Product Builder based in London.*
