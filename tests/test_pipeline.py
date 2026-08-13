"""
Tests for the Q&A Pipeline.

Run: pytest tests/ -v
"""

import os
import pytest
from transformers import pipeline as hf_pipeline

from src.knowledge_base import build_knowledge_base
from src.pipeline import ask_question, get_llm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="module")
def vector_store():
    """Build the vector store once for all tests."""
    return build_knowledge_base(DATA_DIR)


@pytest.fixture(scope="module")
def llm():
    """Load the LLM once for all tests."""
    return get_llm()


# ────────────────────────────────
# ask_question return structure
# ────────────────────────────────
class TestAskQuestionStructure:
    def test_returns_dict(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result, dict), "ask_question should return a dict"

    def test_has_answer_key(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert "answer" in result, "Result dict must have an 'answer' key"

    def test_has_sources_key(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert "sources" in result, "Result dict must have a 'sources' key"

    def test_answer_is_string(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result["answer"], str), "'answer' should be a string"
        assert len(result["answer"].strip()) > 0, "'answer' should not be empty"

    def test_sources_is_list(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What services do you offer?")
        assert isinstance(result["sources"], list), "'sources' should be a list"
        assert len(result["sources"]) > 0, "'sources' should not be empty"


# ────────────────────────────────
# Retrieval quality
# ────────────────────────────────
class TestRetrieval:
    def test_retrieves_pricing_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "How much does the Growth package cost?")
        sources_text = " ".join(result["sources"]).lower()
        assert "growth" in sources_text or "$5,500" in sources_text, (
            "Sources should contain pricing-related content"
        )

    def test_retrieves_seo_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Do you offer SEO services?")
        sources_text = " ".join(result["sources"]).lower()
        assert "seo" in sources_text or "keyword" in sources_text, (
            "Sources should contain SEO-related content"
        )

    def test_retrieves_branding_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Do you help with logo design and branding?")
        sources_text = " ".join(result["sources"]).lower()
        assert "brand" in sources_text or "logo" in sources_text, (
            "Sources should contain branding-related content"
        )

    def test_retrieves_email_marketing_info(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What email marketing platforms do you use?")
        sources_text = " ".join(result["sources"]).lower()
        assert "email" in sources_text or "klaviyo" in sources_text or "mailchimp" in sources_text, (
            "Sources should contain email marketing-related content"
        )

    def test_different_questions_get_different_sources(self, vector_store, llm):
        r1 = ask_question(vector_store, llm, "How does onboarding work?")
        r2 = ask_question(vector_store, llm, "What are your PPC management fees?")
        assert r1["sources"] != r2["sources"], (
            "Different questions should retrieve different chunks"
        )


# ────────────────────────────────
# Answer generation
# ────────────────────────────────
class TestAnswerGeneration:
    def test_answer_is_not_just_the_prompt(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Can I cancel my contract?")
        assert "Context:" not in result["answer"], (
            "Answer should be the generated text, not the full prompt"
        )

    def test_answer_responds_to_question(self, vector_store, llm):
        result = ask_question(vector_store, llm, "How much is the Starter package?")
        answer = result["answer"].lower()
        assert "2,500" in answer or "2500" in answer or "starter" in answer, (
            "Answer should address the pricing question"
        )

    def test_answer_addresses_enterprise_pricing(self, vector_store, llm):
        result = ask_question(vector_store, llm, "How much does the Enterprise package cost?")
        answer = result["answer"].lower()
        assert "12,000" in answer or "12000" in answer or "enterprise" in answer, (
            "Answer should address the Enterprise pricing question"
        )


# ────────────────────────────────
# Out-of-scope handling (hallucination guard)
# ────────────────────────────────
class TestOutOfScopeHandling:
    def test_out_of_scope_returns_fallback_message(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What is the capital of France?")
        assert result["answer"] == "I don't have enough information to answer that.", (
            "Out-of-scope questions should return the fallback message instead of a generated answer"
        )

    def test_out_of_scope_does_not_call_llm(self, vector_store):
        def exploding_llm(prompt):
            raise AssertionError("LLM should not be called for out-of-scope questions")

        result = ask_question(vector_store, exploding_llm, "How do I bake a chocolate cake?")
        assert result["answer"] == "I don't have enough information to answer that.", (
            "Out-of-scope questions should be rejected before ever reaching the LLM"
        )

    def test_out_of_scope_still_returns_sources(self, vector_store, llm):
        result = ask_question(vector_store, llm, "Explain quantum entanglement.")
        assert isinstance(result["sources"], list) and len(result["sources"]) > 0, (
            "Sources should still be returned for transparency even when the answer is rejected"
        )

    def test_on_topic_question_is_not_rejected(self, vector_store, llm):
        result = ask_question(vector_store, llm, "What tools do you use for reporting?")
        assert result["answer"] != "I don't have enough information to answer that.", (
            "On-topic questions should not be caught by the out-of-scope guard"
        )
