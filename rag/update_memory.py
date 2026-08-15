"""
update_memory.py
-----------------
FastAPI endpoint module for the Reverse Feedback Loop.

HOW IT WORKS:
  When a Salesforce Case is Closed AND its Origin was 'Triage Layer',
  the Salesforce Apex class (ReverseFeedbackCallout.cls) fires a POST
  to /api/update_memory with the original customer question + the final
  correct category that was resolved by a human/Agentforce agent.

  This endpoint:
    1. Receives the payload (question + resolved_category)
    2. Embeds and stores it directly into ChromaDB via AdaptiveMemory
    3. Returns a confirmation

  From that point on, the RAG Triage Layer will find this new memory
  when similar tickets arrive — raising confidence above 80% and
  reducing escalations automatically.

PAYLOAD SCHEMA:
  POST /api/update_memory
  {
    "question":           "I can't reset my password",
    "resolved_category":  "Account & Password",
    "case_id":            "5001A000001XyZQ",   // optional — for audit trail
    "customer_context":   "Case closed by Agent"  // optional
  }
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from rag.vector_db import AdaptiveMemory
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared AdaptiveMemory instance — imported by main_enhanced.py
_memory: Optional[AdaptiveMemory] = None

def set_memory_instance(memory: AdaptiveMemory):
    """Called once from main_enhanced.py to inject the shared memory instance."""
    global _memory
    _memory = memory


class UpdateMemoryRequest(BaseModel):
    question: str
    resolved_category: str
    case_id: Optional[str] = None          # Salesforce Case ID for audit
    customer_context: Optional[str] = ""   # Extra context if available


class UpdateMemoryResponse(BaseModel):
    success: bool
    message: str
    case_id: Optional[str] = None


@router.post("/api/update_memory", response_model=UpdateMemoryResponse)
async def update_memory(request: UpdateMemoryRequest):
    """
    Reverse Feedback Loop endpoint.

    Called by Salesforce (ReverseFeedbackCallout.cls) whenever a Case is
    Closed with Origin = 'Triage Layer'.  Stores the verified
    question → category mapping into ChromaDB so future similar tickets
    are auto-resolved with higher confidence.
    """
    if _memory is None:
        raise HTTPException(
            status_code=500,
            detail="AdaptiveMemory not initialised. Check server startup."
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="'question' field cannot be empty.")

    if not request.resolved_category.strip():
        raise HTTPException(status_code=400, detail="'resolved_category' field cannot be empty.")

    try:
        context = request.customer_context or ""
        if request.case_id:
            context = f"CaseID:{request.case_id} | {context}".strip(" |")

        _memory.add_correction(
            text=request.question.strip(),
            correct_label=request.resolved_category.strip(),
            customer_context=context
        )

        logger.info(
            f"[ReverseFeedback] Memory updated — "
            f"CaseID: {request.case_id} | "
            f"Category: {request.resolved_category} | "
            f"Text: {request.question[:60]}..."
        )

        return UpdateMemoryResponse(
            success=True,
            message=(
                f"Memory updated successfully. "
                f"Future tickets similar to '{request.question[:50]}...' "
                f"will now be auto-routed to '{request.resolved_category}'."
            ),
            case_id=request.case_id
        )

    except Exception as e:
        logger.error(f"[ReverseFeedback] Failed to update memory: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update ChromaDB memory: {str(e)}"
        )
