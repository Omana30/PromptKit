"""
test_templates.py — Unit tests for PromptKit template structure validation.

These tests validate template structure, formatting, and logic WITHOUT
making real API calls — so they run instantly and cost nothing.

Run tests:
    pytest tests/test_templates.py -v

Part of PromptKit by Omana Prabhakar (github.com/Omana30)
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── EXTRACTION TESTS ─────────────────────────────────────────────────────────

class TestExtractionTemplates:

    def test_extract_entities_imports(self):
        """extraction.py imports without errors."""
        from templates.extraction import (
            extract_entities,
            extract_key_facts,
            extract_action_items,
            extract_medical_terms,
            extract_sentiment_with_reasoning,
            validate_output
        )
        assert callable(extract_entities)
        assert callable(extract_key_facts)
        assert callable(extract_action_items)
        assert callable(extract_medical_terms)
        assert callable(extract_sentiment_with_reasoning)
        assert callable(validate_output)

    def test_validate_output_passes_valid_dict(self):
        """validate_output returns True for valid dict with required keys."""
        from templates.extraction import validate_output
        result = validate_output(
            {"sentiment": "positive", "confidence": 0.9, "reasoning": "test"},
            required_keys=["sentiment", "confidence"]
        )
        assert result is True

    def test_validate_output_raises_on_missing_keys(self):
        """validate_output raises ValueError when required keys are missing."""
        from templates.extraction import validate_output
        with pytest.raises(ValueError) as exc_info:
            validate_output(
                {"sentiment": "positive"},
                required_keys=["sentiment", "confidence", "reasoning"]
            )
        assert "confidence" in str(exc_info.value)
        assert "reasoning" in str(exc_info.value)

    def test_validate_output_passes_list(self):
        """validate_output handles list input gracefully."""
        from templates.extraction import validate_output
        result = validate_output(
            [{"task": "Send report", "owner": "John", "deadline": "Friday"}],
            required_keys=["task", "owner"]
        )
        assert result is True

    def test_parse_json_valid(self):
        """_parse_json correctly parses clean JSON."""
        from templates.extraction import _parse_json
        result = _parse_json('{"name": "Omana", "role": "AI Engineer"}')
        assert result["name"] == "Omana"
        assert result["role"] == "AI Engineer"

    def test_parse_json_strips_markdown_fences(self):
        """_parse_json strips markdown code fences before parsing."""
        from templates.extraction import _parse_json
        raw = '```json\n{"name": "Omana", "role": "AI Engineer"}\n```'
        result = _parse_json(raw)
        assert result["name"] == "Omana"

    def test_parse_json_raises_on_invalid(self):
        """_parse_json raises ValueError on malformed JSON."""
        from templates.extraction import _parse_json
        with pytest.raises(ValueError) as exc_info:
            _parse_json("this is not json at all")
        assert "invalid JSON" in str(exc_info.value)

    def test_extract_entities_function_signature(self):
        """extract_entities has correct parameter types and defaults."""
        import inspect
        from templates.extraction import extract_entities
        sig = inspect.signature(extract_entities)
        params = sig.parameters
        assert "text" in params
        assert "entity_types" in params
        assert "provider" in params
        assert params["provider"].default == "openai"

    def test_extract_key_facts_function_signature(self):
        """extract_key_facts requires text and facts_to_extract."""
        import inspect
        from templates.extraction import extract_key_facts
        sig = inspect.signature(extract_key_facts)
        params = sig.parameters
        assert "text" in params
        assert "facts_to_extract" in params


# ─── SUMMARISATION TESTS ──────────────────────────────────────────────────────

class TestSummarisationTemplates:

    def test_summarisation_imports(self):
        """summarisation.py imports without errors."""
        from templates.summarisation import (
            executive_summary,
            bullet_summary,
            one_line_summary,
            audience_summary,
            progressive_summary,
            validate_summary
        )
        assert callable(executive_summary)
        assert callable(bullet_summary)
        assert callable(one_line_summary)
        assert callable(audience_summary)
        assert callable(progressive_summary)
        assert callable(validate_summary)

    def test_validate_summary_passes_valid_string(self):
        """validate_summary passes for non-empty string."""
        from templates.summarisation import validate_summary
        result = validate_summary("This is a valid summary.", min_length=10)
        assert result is True

    def test_validate_summary_raises_on_short_string(self):
        """validate_summary raises for string below min_length."""
        from templates.summarisation import validate_summary
        with pytest.raises(ValueError):
            validate_summary("Hi", min_length=10)

    def test_validate_summary_passes_valid_list(self):
        """validate_summary passes for non-empty list."""
        from templates.summarisation import validate_summary
        result = validate_summary(["Point one", "Point two"])
        assert result is True

    def test_validate_summary_raises_on_empty_list(self):
        """validate_summary raises for empty list."""
        from templates.summarisation import validate_summary
        with pytest.raises(ValueError):
            validate_summary([])

    def test_executive_summary_signature(self):
        """executive_summary has correct parameters."""
        import inspect
        from templates.summarisation import executive_summary
        sig = inspect.signature(executive_summary)
        params = sig.parameters
        assert "text" in params
        assert "max_sentences" in params
        assert "focus" in params
        assert params["max_sentences"].default == 3

    def test_audience_summary_signature(self):
        """audience_summary defaults to non-technical."""
        import inspect
        from templates.summarisation import audience_summary
        sig = inspect.signature(audience_summary)
        assert sig.parameters["audience"].default == "non-technical"


# ─── STRUCTURED OUTPUT TESTS ──────────────────────────────────────────────────

class TestStructuredOutputTemplates:

    def test_structured_output_imports(self):
        """structured_output.py imports without errors."""
        from templates.structured_output import (
            enforce_json_schema,
            generate_soap_note,
            format_meeting_notes,
            generate_user_story,
            format_bug_report,
            validate_soap_note
        )
        assert callable(enforce_json_schema)
        assert callable(generate_soap_note)
        assert callable(format_meeting_notes)
        assert callable(generate_user_story)
        assert callable(format_bug_report)
        assert callable(validate_soap_note)

    def test_validate_soap_note_passes_complete_note(self):
        """validate_soap_note passes when all sections present."""
        from templates.structured_output import validate_soap_note
        complete_note = {
            "subjective": "Patient reports chest pain for 3 days.",
            "objective": "BP 140/90, HR 88.",
            "assessment": "Likely musculoskeletal chest pain.",
            "plan": "Prescribe ibuprofen 400mg TDS.",
            "follow_up": "Review in 1 week."
        }
        result = validate_soap_note(complete_note)
        assert result is True

    def test_validate_soap_note_raises_on_missing_section(self):
        """validate_soap_note raises ValueError when sections missing."""
        from templates.structured_output import validate_soap_note
        incomplete_note = {
            "subjective": "Patient reports pain.",
            "objective": "BP normal."
        }
        with pytest.raises(ValueError) as exc_info:
            validate_soap_note(incomplete_note)
        assert "assessment" in str(exc_info.value)

    def test_generate_soap_note_signature(self):
        """generate_soap_note accepts transcript and optional patient_context."""
        import inspect
        from templates.structured_output import generate_soap_note
        sig = inspect.signature(generate_soap_note)
        params = sig.parameters
        assert "transcript" in params
        assert "patient_context" in params
        assert params["patient_context"].default is None


# ─── CHAIN OF THOUGHT TESTS ───────────────────────────────────────────────────

class TestChainOfThoughtTemplates:

    def test_cot_imports(self):
        """chain_of_thought.py imports without errors."""
        from templates.chain_of_thought import (
            decompose_problem,
            pros_cons_analysis,
            decision_framework,
            root_cause_analysis,
            step_by_step_wrapper,
            validate_reasoning_output
        )
        assert callable(decompose_problem)
        assert callable(pros_cons_analysis)
        assert callable(decision_framework)
        assert callable(root_cause_analysis)
        assert callable(step_by_step_wrapper)
        assert callable(validate_reasoning_output)

    def test_validate_reasoning_output_passes(self):
        """validate_reasoning_output passes for complete output."""
        from templates.chain_of_thought import validate_reasoning_output
        result = validate_reasoning_output(
            {"answer": "Use ChromaDB", "confidence": 0.85, "reasoning_steps": []},
            required_keys=["answer", "confidence"]
        )
        assert result is True

    def test_validate_reasoning_output_raises_on_missing(self):
        """validate_reasoning_output raises when keys missing."""
        from templates.chain_of_thought import validate_reasoning_output
        with pytest.raises(ValueError):
            validate_reasoning_output(
                {"answer": "Use ChromaDB"},
                required_keys=["answer", "confidence", "reasoning_steps"]
            )

    def test_decision_framework_signature(self):
        """decision_framework requires question and options."""
        import inspect
        from templates.chain_of_thought import decision_framework
        sig = inspect.signature(decision_framework)
        params = sig.parameters
        assert "question" in params
        assert "options" in params
        assert "criteria" in params
        assert params["criteria"].default is None


# ─── RAG QUERY TESTS ──────────────────────────────────────────────────────────

class TestRAGQueryTemplates:

    def test_rag_imports(self):
        """rag_queries.py imports without errors."""
        from templates.rag_queries import (
            rewrite_query,
            hypothetical_document,
            answer_with_citations,
            no_answer_fallback,
            multi_hop_reasoning,
            validate_rag_response
        )
        assert callable(rewrite_query)
        assert callable(hypothetical_document)
        assert callable(answer_with_citations)
        assert callable(no_answer_fallback)
        assert callable(multi_hop_reasoning)
        assert callable(validate_rag_response)

    def test_validate_rag_response_passes(self):
        """validate_rag_response passes for response with answer field."""
        from templates.rag_queries import validate_rag_response
        result = validate_rag_response({
            "answer": "Metformin should be used with caution.",
            "confidence": 0.85,
            "citations": []
        })
        assert result is True

    def test_validate_rag_response_raises_without_answer(self):
        """validate_rag_response raises ValueError when answer missing."""
        from templates.rag_queries import validate_rag_response
        with pytest.raises(ValueError) as exc_info:
            validate_rag_response({"citations": [], "confidence": 0.5})
        assert "answer" in str(exc_info.value)

    def test_validate_rag_response_raises_on_invalid_confidence(self):
        """validate_rag_response raises when confidence out of range."""
        from templates.rag_queries import validate_rag_response
        with pytest.raises(ValueError):
            validate_rag_response({"answer": "test", "confidence": 1.5})

    def test_rewrite_query_signature(self):
        """rewrite_query has correct defaults."""
        import inspect
        from templates.rag_queries import rewrite_query
        sig = inspect.signature(rewrite_query)
        params = sig.parameters
        assert "original_query" in params
        assert "num_variants" in params
        assert params["num_variants"].default == 3

    def test_hypothetical_document_signature(self):
        """hypothetical_document defaults to paragraph length."""
        import inspect
        from templates.rag_queries import hypothetical_document
        sig = inspect.signature(hypothetical_document)
        params = sig.parameters
        assert params["length"].default == "paragraph"
        assert params["document_type"].default == "general"


# ─── INTEGRATION STRUCTURE TESTS ─────────────────────────────────────────────

class TestProjectStructure:

    def test_all_template_files_exist(self):
        """All required template files exist in the templates directory."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        required_files = [
            "templates/extraction.py",
            "templates/summarisation.py",
            "templates/structured_output.py",
            "templates/chain_of_thought.py",
            "templates/rag_queries.py",
        ]
        for filepath in required_files:
            full_path = os.path.join(base, filepath)
            assert os.path.exists(full_path), f"Missing file: {filepath}"

    def test_all_example_files_exist(self):
        """All required example files exist."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        required_files = [
            "examples/run_extraction.py",
            "examples/run_summarisation.py",
            "examples/run_rag.py",
        ]
        for filepath in required_files:
            full_path = os.path.join(base, filepath)
            assert os.path.exists(full_path), f"Missing file: {filepath}"

    def test_requirements_file_exists(self):
        """requirements.txt exists at root."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "requirements.txt")
        assert os.path.exists(path)

    def test_readme_exists(self):
        """README.md exists at root."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "README.md")
        assert os.path.exists(path)