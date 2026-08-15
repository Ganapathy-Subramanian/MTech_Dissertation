"""
Real tests for llm/agent.py. No GROQ/GEMINI key is set in the test
environment, so analyze_ticket() exercises the local rule-based
fallback path deterministically (no network calls).
"""
import pytest
from llm.agent import LLMAgent


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return LLMAgent()


class TestLLMAgentFallback:
    def test_analyze_ticket_returns_required_keys(self, agent):
        result = agent.analyze_ticket(
            "I was billed twice this month", category="Billing & Payments"
        )
        for key in ("summary", "action", "customer_reply", "resolution", "escalate"):
            assert key in result

    def test_analyze_ticket_escalate_is_bool(self, agent):
        result = agent.analyze_ticket("My policy was cancelled by mistake",
                                       category="Policy Changes")
        assert isinstance(result["escalate"], bool)

    def test_emergency_category_escalates(self, agent):
        result = agent.analyze_ticket("This is an emergency, need help now",
                                       category="Emergency Services")
        assert result["escalate"] is True

    def test_billing_category_does_not_escalate(self, agent):
        result = agent.analyze_ticket("Refund please", category="Refund & Returns")
        assert result["escalate"] is False

    def test_get_complex_response_returns_string(self, agent):
        response = agent.get_complex_response("I forgot my password")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_routing_table_has_expected_categories(self, agent):
        assert agent.ROUTING["Claims"] == "Claims Processing Team"
        assert agent.ROUTING["Technical Support"] == "Tech Support"

    def test_build_prompt_includes_ticket_text(self, agent):
        prompt = agent._build_prompt("Sample ticket text", "General Inquiry",
                                      "Medium", "Neutral")
        assert "Sample ticket text" in prompt
        assert "General Inquiry" in prompt

    def test_gemini_alias_is_llm_agent(self):
        from llm.agent import GeminiAgent
        assert GeminiAgent is LLMAgent
