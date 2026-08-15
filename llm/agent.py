"""
llm/agent.py  —  Multi-provider LLM agent
==========================================
Priority:  Groq (Llama 3 70b) → Gemini Flash → Local rule-based fallback
"""

import os, json, re
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMAgent:
    ROUTING = {
        "Claims":                "Claims Processing Team",
        "Policy & Coverage":     "Policy Advisory Team",
        "Billing & Payments":    "Billing Operations",
        "Complaints & Feedback": "Customer Relations — Escalation",
        "Technical Support":     "Tech Support",
        "Policy Changes":        "Policy Management",
        "Emergency Services":    "Emergency Response — IMMEDIATE",
        "General Inquiry":       "General Support",
        "Account & Password":    "Account Services",
        "Refund & Returns":      "Billing Operations",
    }

    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self._gemini_model = None

        if self.groq_key:
            print("[LLMAgent] Using Groq (Llama 3 70b)")
        elif self.gemini_key:
            print("[LLMAgent] Using Gemini Flash (Groq key not set)")
            self._init_gemini()
        else:
            print("[LLMAgent] No API keys — using local rule-based fallback.")

    def _init_gemini(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            for name in ['gemini-2.0-flash', 'gemini-1.5-flash']:
                try:
                    self._gemini_model = genai.GenerativeModel(name)
                    print(f"[LLMAgent] Gemini model: {name}")
                    break
                except Exception:
                    continue
        except Exception as e:
            print(f"[LLMAgent] Gemini init failed: {e}")

    # ── Public API ─────────────────────────────────────────────────────────

    def get_complex_response(self, user_query: str) -> str:
        result = self.analyze_ticket(user_query)
        return result.get("customer_reply", result.get("summary", "Your case has been received."))

    def analyze_ticket(self, ticket: str, category: str = "", priority: str = "Medium",
                       sentiment: str = "Neutral") -> dict:
        prompt = self._build_prompt(ticket, category, priority, sentiment)

        if self.groq_key:
            result = self._call_groq(prompt)
            if result:
                return result

        if self._gemini_model:
            result = self._call_gemini(prompt)
            if result:
                return result

        return self._local_fallback(ticket, category, priority, sentiment)

    # ── Prompt ─────────────────────────────────────────────────────────────

    def _build_prompt(self, ticket, category, priority, sentiment):
        # IMPORTANT: Tell the model to use only plain ASCII in JSON values.
        # Em-dashes, arrows, en-dashes break JSON parsing.
        return f"""You are an expert insurance CRM analyst. Analyze this ticket.

Return ONLY a valid JSON object with exactly these 5 keys. Rules:
- Use ONLY plain ASCII characters (no em-dash, no unicode arrows, use plain hyphen - instead)
- No markdown, no code fences, no explanation outside the JSON
- "summary": one sentence, max 20 words, specific to the issue
- "action": STRING ONLY - write as one string: "1. Step one. 2. Step two. 3. Step three. 4. Step four." - do NOT use a JSON object or array for this field
- "customer_reply": 3 sentences - acknowledge the issue, state what will happen, give a specific timeline (e.g. 3-5 business days)
- "resolution": one sentence describing the resolution path and responsible team
- "escalate": false for billing/refund/account/technical issues, true only for emergency or legal threats - must be boolean not string

Ticket: {ticket}
Category: {category or "Unknown"}
Priority: {priority}
Sentiment: {sentiment}"""

    # ── Providers ──────────────────────────────────────────────────────────

    def _call_groq(self, prompt: str):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.2,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                return self._parse_json(raw)
            else:
                print(f"[Groq] HTTP {resp.status_code}")
        except Exception as e:
            print(f"[Groq] Error: {e}")
        return None

    def _call_gemini(self, prompt: str):
        try:
            resp = self._gemini_model.generate_content(prompt)
            return self._parse_json(resp.text.strip())
        except Exception as e:
            print(f"[Gemini] Error: {e}")
        return None

    def _parse_json(self, raw: str):
        try:
            # Strip markdown fences
            clean = raw.replace("```json", "").replace("```", "").strip()

            # Find the JSON object boundaries
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start < 0 or end <= start:
                print(f"[LLMAgent] No JSON object found in response")
                return None
            clean = clean[start:end]

            # Fix common Unicode characters that LLMs emit inside JSON strings
            # which break the parser when not escaped properly
            replacements = {
                "\u2014": "-",   # em-dash —
                "\u2013": "-",   # en-dash –
                "\u2192": "->",  # arrow →
                "\u2190": "<-",  # left arrow ←
                "\u2018": "'",   # left single quote '
                "\u2019": "'",   # right single quote '
                "\u201c": '"',   # left double quote "  (rare inside JSON)
                "\u201d": '"',   # right double quote "
                "\u2022": "-",   # bullet •
                "\u00e2": "",    # common UTF-8 decode artifact
            }
            for char, replacement in replacements.items():
                clean = clean.replace(char, replacement)

            # Remove any control characters except \n \r \t
            clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', clean)

            parsed = json.loads(clean)

            # Safety: if Groq returned action as a dict/list instead of a string, flatten it
            if isinstance(parsed.get("action"), dict):
                steps = parsed["action"]
                parsed["action"] = " ".join(f"{k}. {v}" for k, v in steps.items())
            elif isinstance(parsed.get("action"), list):
                parsed["action"] = " ".join(f"{i+1}. {s}" for i, s in enumerate(parsed["action"]))

            # Safety: escalate must be bool
            if isinstance(parsed.get("escalate"), str):
                parsed["escalate"] = parsed["escalate"].lower() == "true"

            return parsed

        except json.JSONDecodeError as e:
            print(f"[LLMAgent] JSON parse error: {e}")
            print(f"[LLMAgent] Raw response: {raw[:300]}")
            return None

    # ── Local fallback ─────────────────────────────────────────────────────

    def _local_fallback(self, ticket: str, category: str, priority: str, sentiment: str) -> dict:
        q = ticket.lower()
        routing_team = self.ROUTING.get(category, "General Support")
        negative = sentiment in ("Negative", "Very Negative")
        escalate = priority in ("Critical", "High")

        if "claims" in (category or "").lower() or any(w in q for w in ["claim","accident","damage","flood","fire","hit","payout"]):
            summary = "Customer is requesting to file or urgently follow up on an insurance claim."
            action = "1. Pull customer policy record and verify coverage. 2. Assign a licensed claims adjuster. 3. Send claim reference number within 1 hour. 4. Set 24-hour callback for status update."
            reply = ("We have received your claim request and a licensed adjuster has been assigned. "
                     "You can expect contact within 4 hours and a claim reference number by email. "
                     "Please let us know if you need anything in the meantime.")
            resolution = "Claims adjuster assessment and payout initiation within SLA."

        elif "billing" in (category or "").lower() or any(w in q for w in ["charge","bill","payment","premium","refund","autopay","invoice"]):
            summary = "Customer has a billing discrepancy, duplicate charge, or payment request."
            action = "1. Pull billing history for last 90 days. 2. Identify duplicate or incorrect charge. 3. Initiate refund if confirmed. 4. Send payment confirmation email."
            reply = ("We apologize for the billing inconvenience and our team is reviewing your account now. "
                     "If a duplicate charge is confirmed, it will be reversed within 3-5 business days. "
                     "You will receive a confirmation email once resolved.")
            resolution = "Billing review, refund or correction, and confirmation within 3 business days."

        elif "refund" in (category or "").lower():
            summary = "Customer is requesting a refund for a billing error or overpayment."
            action = "1. Verify payment record and identify overpayment. 2. Calculate refund amount. 3. Process via finance team. 4. Confirm ETA to customer."
            reply = ("We have received your refund request and our billing team is reviewing your account. "
                     "If a refund is applicable, it will be processed within 3-5 business days. "
                     "You will receive an email confirmation once the refund is initiated.")
            resolution = "Finance team processes refund and confirms timeline with customer."

        elif "complaints" in (category or "").lower() or any(w in q for w in ["rude","complaint","escalat","supervisor","awful","terrible"]):
            summary = "Customer has filed a complaint about service quality and requests escalation."
            action = "1. Acknowledge complaint immediately and apologize. 2. Log in CRM as high-priority. 3. Escalate to supervisor. 4. Arrange callback within 2 hours."
            reply = ("We sincerely apologize for your experience and take all complaints very seriously. "
                     "Your case has been escalated to a senior supervisor who will personally contact you within 2 hours. "
                     "We are committed to making this right.")
            resolution = "Supervisor review, formal apology and remediation, case closure."
            escalate = True

        elif "emergency" in (category or "").lower() or any(w in q for w in ["emergency","fire","flood","urgent","immediately","life"]):
            summary = "URGENT: Customer requires immediate emergency coverage activation."
            action = "1. Transfer to emergency hotline immediately. 2. Activate emergency coverage. 3. Log as Critical case. 4. Notify duty manager. 5. Do NOT place on hold."
            reply = ("We are treating your situation as an emergency and activating your emergency coverage now. "
                     "An emergency response agent is being connected to you. "
                     "Please stay on the line.")
            resolution = "Emergency response team activation and immediate coverage."
            escalate = True

        elif "account" in (category or "").lower() or any(w in q for w in ["password","login","access","locked"]):
            summary = "Customer is unable to access their account and needs authentication help."
            action = "1. Verify customer identity. 2. Initiate password reset and send reset link. 3. Confirm access is restored. 4. Enable 2FA if requested."
            reply = ("We are initiating a password reset for your account now. "
                     "Please check your registered email address for the reset link. "
                     "If the email does not arrive within 5 minutes, please check your spam folder.")
            resolution = "Identity verified, password reset issued, access confirmed."

        elif "technical" in (category or "").lower() or any(w in q for w in ["error","crash","portal","app","website","bug"]):
            summary = "Customer is experiencing a technical issue with the portal or application."
            action = "1. Identify the error code or message. 2. Check system status page. 3. Provide workaround if available. 4. Escalate to DevOps if system-wide. 5. Follow up within 4 hours."
            reply = ("We are sorry you are experiencing technical difficulties and our team has been alerted. "
                     "Please try clearing your browser cache as a first step. "
                     "We will follow up with a resolution within 4 hours.")
            resolution = "Tech team diagnosis, fix or workaround deployed, resolution confirmed."

        # If the customer mentions cancelling plus money/refund related words,
        # prefer billing/refund routing before treating it as a policy change.
        elif any(w in q for w in ["cancel","cancelled","cancellation"]) and any(w in q for w in ["refund","money","paid","payout","receive","received","payment"]):
            summary = "Customer cancelled their policy and is requesting refund/payment follow-up."
            action = "1. Verify cancellation date and payment records. 2. Calculate any due refund. 3. Initiate refund via finance if applicable. 4. Notify customer with ETA."
            reply = ("We understand you've cancelled your policy and are awaiting a refund. "
                     "Our billing team will review your account and process any eligible refund. "
                     "You can expect an update within 3-5 business days.")
            resolution = "Finance team reviews cancellation and issues refund if applicable."

        elif "policy changes" in (category or "").lower() or any(w in q for w in ["change","add","remove","switch","modify","update"]):
            summary = "Customer is requesting a modification to their existing insurance policy."
            action = "1. Verify customer identity. 2. Pull current policy details. 3. Confirm requested change and premium impact. 4. Process change and send updated documents."
            reply = (f"Your policy change request has been received and is being processed. "
                     f"Our {routing_team} will confirm changes and any premium adjustments within 24 hours. "
                     "An updated policy document will be emailed to you.")
            resolution = "Policy change processed, premium recalculated, updated documents issued."

        else:
            summary = "Customer requires general support assistance."
            action = f"1. Review customer query in detail. 2. Pull account information. 3. Provide accurate answer or route to {routing_team}. 4. Log interaction in CRM."
            reply = (f"Thank you for reaching out. Your query has been assigned to our {routing_team}. "
                     "A specialist will review your case and respond within 24 hours. "
                     "Please let us know if there is anything else we can help with.")
            resolution = f"Route to {routing_team}, respond within SLA, close on resolution."

        if negative and not escalate:
            reply = "We understand your frustration and sincerely apologize. " + reply

        return {"summary": summary, "action": action, "customer_reply": reply,
                "resolution": resolution, "escalate": escalate}


# Backward-compatible alias
GeminiAgent = LLMAgent


if __name__ == "__main__":
    agent = LLMAgent()
    ticket = "I was charged twice in December and my refund has not arrived after 3 weeks."
    result = agent.analyze_ticket(ticket, category="Billing & Payments", priority="High", sentiment="Negative")
    print(json.dumps(result, indent=2))