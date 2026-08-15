from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from models.triage import TriageModel
from rag.vector_db import AdaptiveMemory
from rag.update_memory import router as update_memory_router, set_memory_instance
from llm.agent import GeminiAgent
import uvicorn
import os

app = FastAPI(title="AI-Powered Intelligent CRM Agent")

# Initialize components
triage = TriageModel()
memory = AdaptiveMemory()
gemini = GeminiAgent()

# ── Tier 3: Register Reverse Feedback Loop endpoint (/api/update_memory) ──
# Called by Salesforce ReverseFeedbackCallout.cls when a Case is Closed.
# Embeds the resolved question→category into ChromaDB for future auto-routing.
app.include_router(update_memory_router)
set_memory_instance(memory)

# API Models
class TicketRequest(BaseModel):
    text: str

class CorrectionRequest(BaseModel):
    text: str
    correct_label: str

# Endpoints
@app.post("/process-ticket")
async def process_ticket(request: TicketRequest):
    query = request.text
    
    # 1. Check Adaptive Memory (RAG)
    mem_label, dist = memory.query_memory(query)
    if mem_label:
        return {
            "source": "Adaptive Memory (RAG)",
            "category": mem_label,
            "confidence": 1.0 - dist,
            "response": f"Our system learned from a similar case: This is a {mem_label} issue."
        }
    
    # 2. Local Triage Layer (Lightweight NLP)
    label, confidence = triage.predict(query)
    
    # ⚠️ STRICT 10-CLASS: ALWAYS use predicted category (one of 10), NEVER "Complex/Contextual"
    # 3. Decision Logic: High Confidence vs Escalation
    if confidence > 0.7:
        return {
            "source": "Lightweight Triage",
            "category": label,  # One of 10 categories
            "confidence": round(float(confidence), 2),
            "response": f"Automated classification: {label}. How else can I help?"
        }
    else:
        # ⚠️ FIXED: Low confidence → escalate to LLM for better response, but KEEP the predicted category
        # (NO escaping to "Complex/Contextual" — every ticket must be one of the 10)
        llm_response = gemini.get_complex_response(query)
        return {
            "source": "Gemini LLM (Escalated)",
            "category": label,  # FIXED: Use predicted category, NOT "Complex/Contextual"
            "confidence": round(float(confidence), 2),  # Report actual confidence
            "escalated": True,  # Track escalation as separate flag
            "response": llm_response
        }

@app.post("/add-correction")
async def add_correction(request: CorrectionRequest):
    memory.add_correction(request.text, request.correct_label)
    return {"message": "Success", "detail": f"Model will now remember '{request.text}' as {request.correct_label}"}

# Serve Frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    # Ensure static directory exists
    if not os.path.exists("static"):
        os.makedirs("static")
    uvicorn.run(app, host="0.0.0.0", port=8000)
