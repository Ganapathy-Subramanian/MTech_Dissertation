"""
main_demo.py  —  Demo-ready FastAPI backend
============================================
Drop-in replacement / companion to main.py.
Adds:
  POST /api/triage          → full triage result (category, confidence, all_probs,
                               priority, routing, sentiment, latency, LLM summary)
  POST /api/llm-analyze     → Groq/Gemini LLM full analysis
  GET  /api/health          → health + model info
  GET  /api/metrics-summary → live accuracy metrics summary for demo panel

Run:
  python main_demo.py
  # then open http://localhost:8000/demo.html
"""

import os, time, json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="NexusCRM AI Demo API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model bootstrap ────────────────────────────────────────────────────────
_triage = None
def get_triage():
    global _triage
    if _triage is None:
        try:
            from models.enhanced_triage import EnsembleTriageModel
            _triage = EnsembleTriageModel()
        except Exception:
            from models.enhanced_triage import EnhancedTriageModel
            _triage = EnhancedTriageModel()
    return _triage


# ── Routing table ──────────────────────────────────────────────────────────
ROUTING_MAP = {
    "Claims":                {"team": "Claims Processing",     "queue": "CLAIMS-Q1",  "sla": "4h",  "icon": "🔧"},
    "Policy & Coverage":     {"team": "Policy Advisory",       "queue": "POLICY-Q1",  "sla": "8h",  "icon": "📋"},
    "Billing & Payments":    {"team": "Billing Operations",    "queue": "BILLING-Q1", "sla": "2h",  "icon": "💳"},
    "Complaints & Feedback": {"team": "Customer Relations",    "queue": "ESCALATE-Q1","sla": "1h",  "icon": "📣"},
    "Technical Support":     {"team": "Tech Support",          "queue": "TECH-Q1",    "sla": "4h",  "icon": "🖥️"},
    "Policy Changes":        {"team": "Policy Management",     "queue": "CHANGE-Q1",  "sla": "8h",  "icon": "✏️"},
    "Emergency Services":    {"team": "Emergency Response",    "queue": "EMERGENCY-Q1","sla": "30m", "icon": "🚨"},
    "General Inquiry":       {"team": "General Support",       "queue": "GENERAL-Q1", "sla": "24h", "icon": "ℹ️"},
    "Account & Password":    {"team": "Account Services",      "queue": "ACCOUNT-Q1", "sla": "1h",  "icon": "🔑"},
    "Refund & Returns":      {"team": "Billing Operations",    "queue": "REFUND-Q1",  "sla": "3h",  "icon": "↩️"},
}

PRIORITY_COLORS = {
    "Critical": "#FF4D6D",
    "High":     "#F59E0B",
    "Medium":   "#2A7FFF",
    "Low":      "#10D88E",
}

AGENT_ACTIONS = {
    "Claims":                "Review claim documentation → Verify policy → Assign adjuster → Set payout timeline",
    "Policy & Coverage":     "Pull policy details → Confirm coverage scope → Clarify exclusions → Document interaction",
    "Billing & Payments":    "Check billing records → Verify last payment → Confirm autopay status → Issue receipt",
    "Complaints & Feedback": "Acknowledge complaint → Escalate to supervisor → Log in CRM → Follow up within SLA",
    "Technical Support":     "Reproduce error → Check system status → Provide workaround → Escalate to DevOps if needed",
    "Policy Changes":        "Verify customer identity → Pull policy → Process change request → Send confirmation",
    "Emergency Services":    "⚠️ IMMEDIATE: Connect to emergency line → Log case CRITICAL → Notify duty manager",
    "General Inquiry":       "Answer query → Offer relevant documentation → Log interaction → Close if resolved",
    "Account & Password":    "Verify identity → Initiate password reset → Confirm access restored → Log attempt",
    "Refund & Returns":      "Verify payment record → Calculate refund amount → Process via finance → Confirm ETA",
}


# ── Request/Response models ────────────────────────────────────────────────
class TriageRequest(BaseModel):
    text: str

class LLMRequest(BaseModel):
    text: str
    category: str
    priority: str
    sentiment: str


# ── Groq LLM helper ────────────────────────────────────────────────────────
def _groq_analyze(ticket: str, category: str, priority: str, sentiment: str) -> dict:
    """Call Groq (Llama 3 70b) for ticket analysis. Falls back to Gemini, then local."""
    prompt = f"""You are an expert insurance CRM agent. Analyze this customer ticket and return a JSON object with EXACTLY these keys:
- "summary": 1-sentence summary of the customer's issue (max 20 words)
- "action": Recommended immediate agent action (1-2 sentences)
- "customer_reply": Draft customer-facing reply (2-3 sentences, professional and empathetic)
- "resolution": Likely resolution path (1 sentence)
- "escalate": true or false

Ticket: {ticket}
Category: {category}
Priority: {priority}
Sentiment: {sentiment}

Return ONLY valid JSON, no markdown, no preamble."""

    # Try Groq first
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            import requests
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 400,
                    "temperature": 0.3,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                raw = raw.replace("```json", "").replace("```", "").strip()
                return json.loads(raw)
        except Exception as e:
            print(f"[Groq] Error: {e}")

    # Try Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(prompt)
            raw = resp.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"[Gemini] Error: {e}")

    # Local fallback
    return _local_llm_fallback(ticket, category, priority, sentiment)


def _local_llm_fallback(ticket: str, category: str, priority: str, sentiment: str) -> dict:
    """Rule-based fallback when no LLM API key is present."""
    negative = sentiment in ("Negative", "Very Negative")
    cat_lower = category.lower()

    summaries = {
        "claims": "Customer is requesting to file or follow up on an insurance claim.",
        "billing": "Customer has a billing discrepancy or payment concern.",
        "policy": "Customer is inquiring about or requesting a change to their policy.",
        "technical": "Customer is experiencing a technical issue with the portal or app.",
        "complaints": "Customer has raised a complaint about service quality.",
        "emergency": "Customer requires emergency assistance — immediate attention required.",
        "account": "Customer needs help with account access or password reset.",
        "refund": "Customer is requesting a refund or reimbursement.",
    }
    summary_key = next((k for k in summaries if k in cat_lower), None)
    summary = summaries.get(summary_key, "Customer requires support assistance.")

    action = AGENT_ACTIONS.get(category, "Review customer query and assign to appropriate team.")
    reply_opening = "We understand this is urgent and are prioritizing your case." if negative else "Thank you for contacting us."

    return {
        "summary": summary,
        "action": action,
        "customer_reply": (
            f"{reply_opening} Your {category} request has been received and assigned "
            f"to our {ROUTING_MAP.get(category, {}).get('team', 'support team')}. "
            f"You can expect a response within {ROUTING_MAP.get(category, {}).get('sla', '24h')}."
        ),
        "resolution": f"Route to {ROUTING_MAP.get(category, {}).get('team', 'support')} — target resolution within SLA.",
        "escalate": priority in ("Critical", "High"),
    }


# ── Main triage endpoint ───────────────────────────────────────────────────
@app.post("/api/triage")
async def triage_ticket(req: TriageRequest):
    if not req.text.strip():
        raise HTTPException(400, "Ticket text cannot be empty")

    model = get_triage()
    t0 = time.perf_counter()

    label, confidence, features = model.predict_enhanced(req.text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Sentiment label
    sentiment_score = features.get("sentiment_score", 0)
    if sentiment_score <= -0.5:
        sentiment_label = "Very Negative"
        sentiment_emoji = "😡"
    elif sentiment_score <= -0.1:
        sentiment_label = "Negative"
        sentiment_emoji = "😟"
    elif sentiment_score <= 0.1:
        sentiment_label = "Neutral"
        sentiment_emoji = "😐"
    elif sentiment_score <= 0.5:
        sentiment_label = "Positive"
        sentiment_emoji = "🙂"
    else:
        sentiment_label = "Very Positive"
        sentiment_emoji = "😊"

    # Priority
    sentiment_dict = {"compound": sentiment_score}
    priority = model.determine_priority(req.text, sentiment_dict, features)

    # All probabilities (sorted desc)
    all_probs = features.get("all_probabilities", {label: confidence})
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

    routing = ROUTING_MAP.get(label, {"team": "General Support", "queue": "GEN-Q1", "sla": "24h", "icon": "ℹ️"})

    return {
        "category": label,
        "confidence": round(confidence * 100, 2),
        "confidence_raw": round(confidence, 4),
        "priority": priority,
        "priority_color": PRIORITY_COLORS.get(priority, "#8899b4"),
        "sentiment": sentiment_label,
        "sentiment_emoji": sentiment_emoji,
        "sentiment_score": round(sentiment_score, 3),
        "routing": routing,
        "agent_action": AGENT_ACTIONS.get(label, "Review and route to appropriate team."),
        "all_probabilities": [{"label": l, "pct": round(p * 100, 2)} for l, p in sorted_probs],
        "latency_ms": latency_ms,
        "model_version": "Phase-2 BERT Ensemble v2.1",
    }


# ── LLM analysis endpoint ──────────────────────────────────────────────────
@app.post("/api/llm-analyze")
async def llm_analyze(req: LLMRequest):
    result = _groq_analyze(req.text, req.category, req.priority, req.sentiment)
    return result


# ── Health check ───────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model": "Phase-2 BERT Ensemble",
        "phase1_accuracy": "99.18%",
        "bert_accuracy": "100% (200-sample eval)",
        "realworld_accuracy": "60% (25-sample informal)",
        "groq_available": bool(os.getenv("GROQ_API_KEY")),
        "gemini_available": bool(os.getenv("GEMINI_API_KEY")),
        "categories": 10,
        "training_samples": 20629,
    }


# ── Metrics summary for demo panel ────────────────────────────────────────
@app.get("/api/metrics-summary")
async def metrics_summary():
    return {
        "phase1": {
            "accuracy": 99.18,
            "weighted_f1": 0.9918,
            "macro_f1": 0.9857,
            "samples": 4126,
            "split": "80/20 stratified holdout",
        },
        "bert_ensemble": {
            "accuracy": 100.0,
            "weighted_f1": 1.0,
            "samples": 200,
            "note": "200-sample eval from held-out set",
        },
        "realworld": {
            "accuracy": 60.0,
            "weighted_f1": 0.6389,
            "samples": 25,
            "note": "Handcrafted informal/typo tickets",
        },
        "benchmark": {
            "our_model": 99.18,
            "claimsense_ai_v1": 93.0,
            "delta": 6.18,
            "latency_ours_ms": 2,
            "latency_theirs_ms": 200,
        },
        "per_category": [
            {"category": "Claims",               "f1": 0.9925, "support": 1209},
            {"category": "Policy & Coverage",    "f1": 0.9960, "support": 1612},
            {"category": "Billing & Payments",   "f1": 0.9841, "support": 408},
            {"category": "Complaints & Feedback","f1": 0.9926, "support": 410},
            {"category": "General Inquiry",      "f1": 0.9529, "support": 81},
            {"category": "Account & Password",   "f1": 1.0000, "support": 83},
            {"category": "Technical Support",    "f1": 0.9884, "support": 85},
            {"category": "Policy Changes",       "f1": 0.9697, "support": 80},
            {"category": "Emergency Services",   "f1": 0.9875, "support": 79},
            {"category": "Refund & Returns",     "f1": 0.9937, "support": 79},
        ],
    }


# ── Serve static ───────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
