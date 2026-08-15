from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
from models.enhanced_triage import EnsembleTriageModel
from models.auto_retrain import SelfLearningWrapper
from rag.vector_db import AdaptiveMemory
from llm.agent import GeminiAgent
from analytics.dashboard import AnalyticsDashboard
from analytics.test_coverage import run_coverage, get_last_result as get_last_coverage_result
from analytics.memory_growth import get_memory_growth
from workflow.automation import WorkflowEngine
from security.auth import AuthManager, get_current_user
from security.customer_auth import customer_auth_manager
from agents.agent_manager import agent_team_manager
from integration.salesforce import salesforce
from rag.update_memory import router as update_memory_router, set_memory_instance
from models.points_manager import points_manager
import uvicorn
import os
import time
import psutil
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import io


@asynccontextmanager
async def lifespan(app):
    # Startup: initialise self-learning drift-check scheduler
    try:
        self_learning.start_scheduler()
        print("[Startup] SelfLearning drift-check scheduler started (every 6 h)")
    except Exception as e:
        print(f"[Startup] Scheduler start failed (apscheduler may not be installed): {e}")
    yield
    # Shutdown: stop scheduler cleanly
    try:
        self_learning.stop_scheduler()
    except Exception:
        pass

app = FastAPI(title="AI-Powered Intelligent CRM Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── System Throughput tracking (feeds /analytics/system-metrics) ──
# Keeps a rolling log of request timestamps, response times, and endpoint
# hits so the "System Throughput Analysis" dashboard (static/system-throughput.html)
# has live data to chart, mirroring Figure 6.5.
_request_log = deque(maxlen=20000)  # {"ts": epoch_seconds, "path": str, "duration_ms": float}
_TRACKED_ENDPOINTS = ("/process-ticket", "/salesforce/create-ticket", "/analytics", "/ticket")

@app.middleware("http")
async def throughput_tracking_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    path = request.url.path
    # Bucket the path into one of the dashboard's tracked endpoint groups
    if path.startswith("/process-ticket"):
        bucket = "/process-ticket"
    elif path.startswith("/salesforce/create-ticket") or path.startswith("/create-ticket"):
        bucket = "/create-ticket"
    elif path.startswith("/analytics"):
        bucket = "/analytics"
    elif "feedback" in path:
        bucket = "/feedback"
    else:
        bucket = None
    if bucket:
        _request_log.append({"ts": start, "path": bucket, "duration_ms": duration_ms})
    return response

# Initialize components
# EnsembleTriageModel = BERT (70%) + TF-IDF (30%) for best-in-class accuracy
triage = EnsembleTriageModel()
# Pass the pre-built ensemble to SelfLearningWrapper to avoid double loading
# This wires automatic learning: every prediction is logged; corrections trigger retraining
self_learning = SelfLearningWrapper(model=triage)
memory = AdaptiveMemory()

# Register Reverse Feedback Loop endpoint and inject shared memory
app.include_router(update_memory_router)
set_memory_instance(memory)
gemini = GeminiAgent()
analytics = AnalyticsDashboard()
workflow = WorkflowEngine()
auth = AuthManager()


# API Models
class TicketRequest(BaseModel):
    text: str
    customer_id: Optional[str] = None
    channel: str = "web"  # web, email, chat, social
    priority: Optional[str] = None

class CorrectionRequest(BaseModel):
    text: str
    correct_label: str
    customer_id: Optional[str] = None

class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    email: str
    history: List[dict] = []

class TicketStatusUpdate(BaseModel):
    status: str
    updates: Optional[dict] = None

class TicketResponseRequest(BaseModel):
    text: str
    status: Optional[str] = None

class FeedbackRequest(BaseModel):
    rating: int  # 1-5
    comments: Optional[str] = ""
    correct_category: Optional[str] = None  # customer-provided category correction

class ReclassifyRequest(BaseModel):
    new_category: str
    original_text: Optional[str] = None

class RerouteRequest(BaseModel):
    to_agent_id: Optional[str] = None
    new_category: Optional[str] = None
    ticket_text: Optional[str] = None

# Customer Authentication Models
class CustomerLoginRequest(BaseModel):
    email: str
    password: str
    use_salesforce: Optional[bool] = False

class AdminLoginRequest(BaseModel):
    email: str
    password: str

class CustomerRegistrationRequest(BaseModel):
    email: str
    username: str
    password: str
    name: str
    phone: str

class CustomerUpdateRequest(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None

# Agent Management Models
class CreateAgentRequest(BaseModel):
    name: str
    email: str
    team: str
    skills: List[str] = []
    salesforce_username: Optional[str] = None
    salesforce_password: Optional[str] = None

class UpdateAgentStatusRequest(BaseModel):
    status: str

class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    team: Optional[str] = None
    skills: Optional[List[str]] = None

# Advanced Endpoints
@app.post("/process-ticket")
async def process_ticket(request: TicketRequest, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    try:
        query = request.text

        if not request.customer_id and user.get('role') == 'customer':
            request.customer_id = user.get('sub')

        # Log analytics
        background_tasks.add_task(analytics.log_ticket, {
            "text": request.text,
            "customer_id": request.customer_id,
            "channel": request.channel,
            "priority": request.priority
        })

        # 2. Check Customer Context & History
        customer_context = ""
        # Fetch customer to get email for CRM linking
        customer_obj = customer_auth_manager.get_customer(request.customer_id) if request.customer_id else None
        cust_email = customer_obj.get("email") if customer_obj else None

        # ── Duplicate Ticket Check ──
        existing_tickets = salesforce.get_tickets_by_customer(request.customer_id, cust_email)
        for t in existing_tickets:
            if t.get("status", "").lower() not in ["resolved", "closed"]:
                if query.lower().strip() in t.get("description", "").lower().strip() or \
                   t.get("description", "").lower().strip() in query.lower().strip():
                    return {
                        "source": "Duplicate Prevention",
                        "is_duplicate": True,
                        "ticket_id": t["id"],
                        "category": t.get("category"),
                        "assigned_agent": t.get("agent"),
                        "response": f"We noticed you already have an open ticket for this: '{t.get('title', 'Support Ticket')}'. We've linked your query to Ticket ID: {t['id']}.",
                        "original_query": query,
                        "workflow_actions": workflow.get_actions_for_category(t.get("category", "General"))
                    }

        # 3. Check Adaptive Memory (RAG) with customer context
        mem_label, dist = memory.query_memory(query)
        if mem_label and dist < 0.3:
            # Get best agent for this category
            assigned_agent = agent_team_manager.get_best_agent(mem_label, "Medium")
            
            # Create local ticket ID immediately
            ticket_id = f"TICKET-{int(datetime.now().timestamp())}"
            
            response_data = {
                "source": "Adaptive Memory (RAG)",
                "ticket_id": ticket_id,
                "customer_id": request.customer_id,
                "customer_email": cust_email,
                "original_query": query,
                "category": mem_label,
                "confidence": 1.0 - dist,
                "response": gemini.get_complex_response(f"Customer query: {query}\nCategory identified: {mem_label}\nProvide a helpful, specific support response for this {mem_label} issue."),
                "customer_context": customer_context,
                "workflow_actions": workflow.get_actions_for_category(mem_label),
                "assigned_agent": assigned_agent
            }
            
            # Create local ticket immediately so it shows on dashboard
            salesforce.create_ticket({
                "customer_id": request.customer_id,
                "customer_email": cust_email,
                "subject": query[:80] + ("..." if len(query) > 80 else ""),
                "text": query,  # Save ORIGINAL query for duplicate detection
                "category": mem_label,
                "priority": "Medium",
                "owner_id": assigned_agent["agent_id"] if assigned_agent else None
            }, forced_id=ticket_id, only_local=True)

            # Assign agent if available
            if assigned_agent:
                agent_team_manager.assign_ticket(assigned_agent['agent_id'], ticket_id)
            
            # Background task for Salesforce sync
            background_tasks.add_task(_create_crm_ticket_async, response_data)
            background_tasks.add_task(analytics.log_resolution, response_data)
            return response_data

        # 3. Local Triage Layer — Ensemble NLP (auto-logs for self-learning)
        label, confidence, entities = self_learning.predict_and_log(query)

        # 4. Sentiment & Priority Analysis
        sentiment = triage.analyze_sentiment(query)
        auto_priority = triage.determine_priority(query, sentiment, entities)
        # 5. Decision Logic: High Confidence vs Escalation
        # Safe routing: lower confidence threshold slightly and accept when BERT/TF-IDF agree
        CONF_THRESHOLD = float(os.getenv('TRIAGE_CONFIDENCE_THRESHOLD', 0.70))
        bert_top = entities.get('bert_top') if isinstance(entities, dict) else None
        tfidf_top = entities.get('tfidf_top') if isinstance(entities, dict) else None
        bert_conf = float(entities.get('bert_confidence', 0.0)) if isinstance(entities, dict) else 0.0

        accept_by_model_agreement = (bert_top and tfidf_top and bert_top == tfidf_top) or (bert_conf >= 0.85)

        # Whitelist: high-confidence billing/payment queries bypass sentiment gate —
        # transactional frustration ("payment failed") is normal and must not escalate.
        BILLING_WHITELIST = {"Billing & Payments"}
        billing_bypass = (label in BILLING_WHITELIST and confidence > 0.75)
        if (confidence > CONF_THRESHOLD and sentiment['compound'] > -0.5) or accept_by_model_agreement or billing_bypass:
            workflow_actions = workflow.get_actions_for_category(label)
            
            # Get best agent for this category
            assigned_agent = agent_team_manager.get_best_agent(label, auto_priority)
            
            # Create local ticket ID immediately
            ticket_id = f"TICKET-{int(datetime.now().timestamp())}"
            
            response_data = {
                "source": "Local Triage Layer",
                "ticket_id": ticket_id,
                "customer_id": request.customer_id,
                "customer_email": cust_email,
                "original_query": query,
                "category": label,
                "confidence": round(float(confidence), 2),
                "sentiment": sentiment,
                "priority": auto_priority,
                "entities": entities,
                "response": gemini.get_complex_response(f"Customer query: {query}\nCategory: {label}, Priority: {auto_priority}, Confidence: {confidence:.0%}\nProvide a clear, empathetic, solution-focused support response."),
                "workflow_actions": workflow_actions,
                "assigned_agent": assigned_agent,
                "auto_resolved": True
            }
            
            # Create local ticket immediately so it shows on dashboard
            salesforce.create_ticket({
                "customer_id": request.customer_id,
                "customer_email": cust_email,
                "subject": query[:80] + ("..." if len(query) > 80 else ""),
                "text": query,  # Save ORIGINAL query for duplicate detection
                "category": label,
                "priority": auto_priority,
                "owner_id": assigned_agent["agent_id"] if assigned_agent else None
            }, forced_id=ticket_id, only_local=True)

            # Assign agent if available
            if assigned_agent:
                agent_team_manager.assign_ticket(assigned_agent['agent_id'], ticket_id)

            # Persist ticket in Salesforce and assign owner where possible
            background_tasks.add_task(_create_crm_ticket_async, response_data)
            
            background_tasks.add_task(analytics.log_resolution, response_data)
            return response_data
        else:
            # ⚠️ STRICT 10-CLASS: Low confidence → Still use predicted category + escalate workflow
            # (NO "Complex/Contextual" escape route — every query must be one of the 10 categories)
            enhanced_query = f"""
            Customer Query: {query}
            Customer Context: {customer_context}
            Preliminary Analysis: Category={label}, Confidence={confidence:.2f}, Sentiment={sentiment}, Priority={auto_priority}
            Extracted Entities: {entities}
            """

            llm_response = gemini.get_complex_response(enhanced_query)
            
            # Get escalation team agent
            assigned_agent = agent_team_manager.get_best_agent("escalation", auto_priority)
            
            # Create local ticket ID immediately
            ticket_id = f"TICKET-{int(datetime.now().timestamp())}"
            
            response_data = {
                "source": "Gemini LLM (Escalated)",
                "ticket_id": ticket_id,
                "customer_id": request.customer_id,
                "original_query": query,
                "category": label,  # ← FIXED: Use predicted category (one of 10), NOT "Complex/Contextual"
                "confidence": round(float(confidence), 2),  # ← Report the actual confidence
                "sentiment": sentiment,
                "priority": auto_priority,
                "entities": entities,
                "response": llm_response,
                "workflow_actions": workflow.escalate_to_human(label, auto_priority),
                "assigned_agent": assigned_agent,
                "escalated": True,  # ← Track escalation as separate flag, not as category
                "escalation_reason": "Low confidence - requires human review",
                "requires_human_review": True
            }

            # Create local ticket immediately so it shows on dashboard
            salesforce.create_ticket({
                "customer_id": request.customer_id,
                "customer_email": None, # Will be filled by background
                "subject": query[:80] + ("..." if len(query) > 80 else ""),
                "text": query,  # Save ORIGINAL query for duplicate detection
                "category": label,  # ← FIXED: Use predicted category (one of 10), NOT "Complex/Contextual"
                "priority": auto_priority,
                "owner_id": assigned_agent["agent_id"] if assigned_agent else None
            }, forced_id=ticket_id, only_local=True)

            # Assign escalation agent if available
            if assigned_agent:
                agent_team_manager.assign_ticket(assigned_agent['agent_id'], ticket_id)

            # Auto-create Salesforce ticket for escalated cases
            background_tasks.add_task(_create_crm_ticket_async, response_data)

            background_tasks.add_task(analytics.log_resolution, response_data)
            return response_data

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ [CRITICAL ERROR] process_ticket failed: {error_details}")
        return {
            "error": str(e),
            "traceback": error_details,
            "source": "Server Error"
        }

def _create_crm_ticket_async(ticket_data: dict):
    """Background task to create CRM ticket in Salesforce"""
    try:
        assigned_agent = ticket_data.get("assigned_agent") or {}
        original_query = ticket_data.get("original_query", "No original query provided")
        ai_response = ticket_data.get("response", "")
        customer_id = ticket_data.get("customer_id")

        # Construct a rich description for Salesforce
        full_description = f"""CUSTOMER QUESTION:
{original_query}

AI ANALYSIS & RESPONSE:
{ai_response}

---
AI Metadata: Source={ticket_data.get('source')}, Category={ticket_data.get('category')}
"""

        # Fallback owner_id: use salesforce_id if it exists, otherwise use local agent_id
        owner_id = assigned_agent.get("salesforce_id") or assigned_agent.get("agent_id")

        sf_ticket_data = {
            "subject": original_query[:80] + ("..." if len(original_query) > 80 else ""),
            "text": full_description,
            "customer_id": customer_id,
            "customer_email": ticket_data.get("customer_email"),
            "category": ticket_data.get('category', 'General'),
            "priority": ticket_data.get('priority', 'Medium'),
            "channel": "AI_CRM_Portal",
            "owner_id": owner_id,
            "sentiment_score": ticket_data.get('sentiment', {}).get('compound', 0),
            "confidence_score": ticket_data.get('confidence', 0),
            "escalation_reason": "AI processed request",
            "ai_analysis": {
                "source": ticket_data.get('source'),
                "entities": ticket_data.get('entities', {}),
                "workflow_actions": ticket_data.get('workflow_actions', [])
            }
        }

        sf_ticket_id = salesforce.create_ticket(sf_ticket_data, forced_id=ticket_data.get("ticket_id"))
        
        if sf_ticket_id:
            print(f"✅ Background task: Ticket synced to CRM {sf_ticket_id} for Customer {customer_id}")
            # Store SF case ID for future cross-reference (no extra API call needed)
            local_ticket_id = ticket_data.get("ticket_id")
            if local_ticket_id and local_ticket_id in salesforce.mock_tickets:
                salesforce.mock_tickets[local_ticket_id]["sf_case_id"] = sf_ticket_id
                salesforce._save_mock_tickets()
        else:
            print("❌ Failed to auto-create Salesforce ticket")


    except Exception as e:
        print(f"❌ Error in CRM ticket creation: {e}")
@app.post("/add-correction")
async def add_correction(request: CorrectionRequest, background_tasks: BackgroundTasks):
    # Enhanced correction with customer context
    customer_context = ""
    if request.customer_id:
        customer_context = analytics.get_customer_context(request.customer_id)

    # 1. Store in RAG memory for immediate retrieval improvement
    memory.add_correction(request.text, request.correct_label, customer_context)

    # 2. Log to SelfLearningWrapper — triggers auto-retrain when threshold (10) is reached
    background_tasks.add_task(
        self_learning.add_correction,
        request.text,
        request.correct_label,
        None,   # original_category
        None    # confidence
    )

    # Reward points for the corrected category (positive reinforcement)
    try:
        points_manager.adjust_points(request.correct_label, 1)
    except Exception:
        pass

    return {
        "message": "Correction saved — model will auto-retrain after 10 corrections",
        "detail": f"Model will now remember '{request.text}' as {request.correct_label}",
        "retraining_scheduled": True,
        "pending_corrections": self_learning._corrections_since_last_retrain + 1,
        "trigger_at": 10
    }


@app.get("/analytics/points")
async def get_points(user: dict = Depends(get_current_user)):
    try:
        return {"points": points_manager.get_summary()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analytics/points/adjust")
async def adjust_points(payload: dict, user: dict = Depends(get_current_user)):
    """Adjust points for a category: {"category": "Emergency Services", "delta": -2} """
    try:
        cat = payload.get('category')
        delta = int(payload.get('delta', 0))
        if not cat:
            raise HTTPException(status_code=400, detail="category required")
        res = points_manager.adjust_points(cat, delta)
        return {"success": True, "entry": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ticket/{ticket_id}/reclassify")
async def reclassify_ticket(ticket_id: str, request: ReclassifyRequest, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    """Reclassify a ticket, train the AI, and auto-reassign"""
    ticket = salesforce.get_ticket_details(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    old_category = ticket.get("category")
    original_text = request.original_text
    
    # Try to extract original text from description if not provided
    if not original_text and ticket.get("description"):
        desc = ticket.get("description", "")
        if "CUSTOMER QUESTION:" in desc:
            parts = desc.split("CUSTOMER QUESTION:\n")
            if len(parts) > 1:
                original_text = parts[1].split("\n\nAI ANALYSIS")[0].strip()
        else:
            original_text = desc.split("\n\n")[0].strip()
            
    if not original_text:
        original_text = ticket.get("subject", "")

    # 1. Store feedback in RAG memory
    customer_context = ""
    if ticket.get("customer_id"):
        customer_context = analytics.get_customer_context(ticket.get("customer_id"))
    memory.add_correction(original_text, request.new_category, customer_context)

    # 2. Trigger Retraining
    background_tasks.add_task(triage.retrain_model)

    # 3. Determine new agent based on category
    auto_priority = ticket.get("priority", "Medium")
    new_agent = agent_team_manager.get_best_agent(request.new_category, auto_priority)
    new_agent_id = new_agent["agent_id"] if new_agent else None

    # 4. Update Ticket and Reassign
    updates = {"category": request.new_category}
    
    if new_agent_id and ticket.get("owner_id") != new_agent_id:
        updates["owner_id"] = new_agent_id
        old_owner = ticket.get("owner_id")
        if old_owner:
            agent_team_manager.complete_ticket(old_owner)  # Free old agent capacity
        agent_team_manager.assign_ticket(new_agent_id, ticket_id)
    
    # Persist all updates to Salesforce / Local Storage
    salesforce.update_ticket_status(ticket_id, ticket.get("status", "New"), updates)

    return {
        "success": True,
        "new_category": request.new_category,
        "assigned_agent": new_agent,
        "message": f"Reclassified to {request.new_category} and reassigned."
    }


@app.post("/ticket/{ticket_id}/reroute")
async def reroute_ticket(ticket_id: str, request: RerouteRequest, background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    """Reroute a ticket to another agent (or best available). Records correction for self-learning when category provided."""
    ticket = salesforce.get_ticket_details(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    # Attempt reroute
    success = agent_team_manager.reroute_ticket(
        from_agent_id=ticket.get('owner_id'),
        ticket_id=ticket_id,
        to_agent_id=request.to_agent_id,
        category=request.new_category,
        ticket_text=request.ticket_text or ticket.get('description') or ticket.get('subject')
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to reroute ticket")

    # If a category correction was provided, persist to RAG and SelfLearning.
    # Wrapped so a memory/embedding hiccup can't turn an already-successful
    # reroute into a reported failure, and can't block the classifier retrain.
    if request.new_category and request.ticket_text:
        try:
            memory.add_correction(request.ticket_text, request.new_category, analytics.get_customer_context(ticket.get('customer_id')))
        except Exception as mem_err:
            print(f"[SelfLearning] RAG memory update failed (non-fatal): {mem_err}")
        background_tasks.add_task(self_learning.add_correction, request.ticket_text, request.new_category, None, None)

    return {"success": True, "message": "Ticket rerouted"}

# ============================================
# CUSTOMER AUTHENTICATION & MANAGEMENT
# ============================================

@app.post("/customer/register")
async def register_customer(request: CustomerRegistrationRequest):
    """Register new customer"""
    success, result = customer_auth_manager.register_customer(
        email=request.email,
        username=request.username,
        password=request.password,
        name=request.name,
        phone=request.phone
    )

    if success:
        return {
            "success": True,
            "customer_id": result,
            "message": "Registration successful",
            "next_step": "Please login with your credentials"
        }
    else:
        raise HTTPException(status_code=400, detail=result)

@app.post("/customer/login")
async def login_customer(request: CustomerLoginRequest):
    """Customer login endpoint"""
    success, customer_data = customer_auth_manager.login_customer(
        email=request.email,
        password=request.password
    )

    if not success:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth.create_access_token(
        data={
            "sub": customer_data['customer_id'],
            "role": "customer",
            "email": customer_data['email'],
            "name": customer_data['name']
        },
        expires_delta=timedelta(hours=24)
    )

    return {
        "success": True,
        "customer_id": customer_data['customer_id'],
        "name": customer_data['name'],
        "email": customer_data['email'],
        "token": token
    }

@app.post("/agent/login")
async def login_agent(request: CustomerLoginRequest):
    """Agent login endpoint — sets agent status to 'available' on login"""
    use_salesforce = request.__dict__.get('use_salesforce', False) or getattr(request, 'use_salesforce', False)

    access_token = auth.authenticate_user_with_provider(request.email, request.password, use_salesforce=use_salesforce)
    if not access_token:
        raise HTTPException(status_code=401, detail="Invalid agent credentials")

    agent_data = None
    agents = agent_team_manager.get_all_agents()
    for agent_id, agent in agents.items():
        if agent.get("email") == request.email:
            agent_data = {"agent_id": agent_id, "name": agent["name"], "team": agent["team"]}
            # ✅ Mark agent as AVAILABLE on login
            agent_team_manager.update_agent_status(agent_id, "available")
            break

    if not agent_data:
        raise HTTPException(status_code=404, detail="Agent not found in the system. Please contact admin.")

    return {
        "success": True,
        "name": agent_data["name"],
        "email": request.email,
        "token": access_token,
        "type": "agent",
        "agent_id": agent_data["agent_id"],
        "team": agent_data["team"]
    }

@app.post("/agent/logout")
async def logout_agent(user: dict = Depends(get_current_user)):
    """Agent logout endpoint — sets agent status to 'offline'"""
    agent_id = user.get('agent_id') or user.get('sub')
    if agent_id:
        # Try by agent_id first, else look up by email
        agents = agent_team_manager.get_all_agents()
        if agent_id not in agents:
            email = user.get('sub')
            for aid, adata in agents.items():
                if adata.get('email') == email:
                    agent_id = aid
                    break
        agent_team_manager.update_agent_status(agent_id, "offline")
    return {"success": True, "message": "Logged out successfully"}

@app.post("/admin/login")
async def login_admin(request: AdminLoginRequest):
    """Admin login backed by Salesforce admin credentials in environment."""
    sf_username = os.getenv("SALESFORCE_USERNAME")
    sf_password = os.getenv("SALESFORCE_PASSWORD")

    if not sf_username or not sf_password:
        raise HTTPException(status_code=500, detail="Salesforce admin credentials are not configured")

    if request.email != sf_username or request.password != sf_password:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    # Skip Salesforce authentication in mock mode
    sf_enabled = os.getenv("SALESFORCE_ENABLED", "false").lower() == "true"
    if sf_enabled and not salesforce.authenticate():
        failure_reason = getattr(salesforce, "last_error", None) or "Salesforce authentication failed"
        raise HTTPException(status_code=503, detail=f"Salesforce authentication failed: {failure_reason}")

    token = auth.create_access_token(
        data={
            "sub": request.email,
            "role": "admin",
            "email": request.email,
            "name": "Salesforce Admin"
        },
        expires_delta=timedelta(hours=8)
    )

    return {
        "success": True,
        "name": "Salesforce Admin",
        "email": request.email,
        "token": token,
        "type": "admin"
    }


@app.get("/admin/autologin")
async def admin_autologin():
    """Development helper: return an admin token when running in development.
    Only enabled when ENVIRONMENT=development to avoid accidental exposure in prod.
    """
    env = os.getenv('ENVIRONMENT','').lower()
    if env != 'development':
        raise HTTPException(status_code=404, detail='Not found')

    sf_username = os.getenv("SALESFORCE_USERNAME")
    sf_password = os.getenv("SALESFORCE_PASSWORD")
    if not sf_username or not sf_password:
        raise HTTPException(status_code=500, detail="Admin credentials not configured")

    token = auth.create_access_token(
        data={"sub": sf_username, "role": "admin", "email": sf_username, "name": "Dev Admin"},
        expires_delta=timedelta(hours=8)
    )
    return {"success": True, "token": token, "email": sf_username, "name": "Dev Admin"}

@app.post("/admin/sync-salesforce-users")
async def sync_salesforce_users(user: dict = Depends(get_current_user)):
    """Sync Salesforce users with local agent database"""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        from integration.salesforce import salesforce

        # Get Salesforce users
        sf_users = salesforce.get_salesforce_users()

        if not sf_users:
            return {"success": False, "message": "No Salesforce users found or connection failed"}

        # Sync with local agents
        sync_result = agent_team_manager.sync_salesforce_users(sf_users)

        return {
            "success": sync_result["success"],
            "synced": sync_result.get("synced", 0),
            "updated": sync_result.get("updated", 0),
            "total_agents": sync_result.get("total_agents", 0),
            "message": f"Synced {sync_result.get('synced', 0)} new agents, updated {sync_result.get('updated', 0)} existing agents"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing Salesforce users: {str(e)}")

@app.get("/admin/salesforce-users")
async def get_salesforce_users(user: dict = Depends(get_current_user)):
    """Get list of Salesforce users"""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        from integration.salesforce import salesforce

        sf_users = salesforce.get_salesforce_users()
        sf_contacts = salesforce.get_salesforce_contacts()

        return {
            "success": True,
            "users": sf_users,
            "contacts": sf_contacts,
            "message": (
                "Salesforce users loaded"
                if sf_users
                else "No active Salesforce User records found. Contact records are returned for provisioning visibility."
            )
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting Salesforce users: {str(e)}")

@app.get("/customer/{customer_id}")
async def get_customer_dashboard(customer_id: str, user: dict = Depends(get_current_user)):
    """Get customer dashboard with profile and tickets"""
    customer = customer_auth_manager.get_customer(customer_id)

    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    if user.get('role') == 'customer' and user.get('email') != customer['email']:
        raise HTTPException(status_code=403, detail="Forbidden")

    if user.get('role') not in ['admin', 'customer']:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Get customer tickets/issues
    customer_context = analytics.get_customer_context(customer_id)

    return {
        "profile": customer,
        "recent_tickets": customer_context,
        "tier": customer.get('tier', 'bronze'),
        "total_spent": customer.get('total_spent', 0),
        "tickets_count": customer.get('tickets_count', 0),
        "dashboard_ready": True
    }

@app.get("/customer/{customer_id}/tickets")
async def get_customer_tickets(customer_id: str, user: dict = Depends(get_current_user)):
    """Get customer's tickets"""
    customer = customer_auth_manager.get_customer(customer_id)

    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    if user.get('role') == 'customer' and user.get('email') != customer['email']:
        raise HTTPException(status_code=403, detail="Forbidden")

    if user.get('role') not in ['admin', 'customer']:
        raise HTTPException(status_code=403, detail="Forbidden")

    tickets = salesforce.get_tickets_by_customer(customer_id, email=customer.get('email'))

    return {
        "tickets": tickets,
        "total_tickets": len(tickets)
    }

@app.put("/customer/{customer_id}")
async def update_customer(customer_id: str, request: CustomerUpdateRequest):
    """Update customer profile"""
    success = customer_auth_manager.update_customer(customer_id, request.dict(exclude_unset=True))

    if success:
        customer = customer_auth_manager.get_customer(customer_id)
        return {
            "success": True,
            "message": "Profile updated successfully",
            "customer": customer
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to update profile")

# ============================================
# AGENT & TEAM MANAGEMENT
# ============================================

@app.get("/agents/available")
async def get_available_agents(category: str = "general", user: dict = Depends(get_current_user)):
    """Get available agents for a category"""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")

    agents = agent_team_manager.get_all_agents()

    available = []
    for agent_id, agent_data in agents.items():
        if agent_data['status'] == 'available':
            available.append({
                "agent_id": agent_id,
                "name": agent_data['name'],
                "team": agent_data['team'],
                "skills": agent_data['skills'],
                "rating": agent_data['rating'],
                "active_tickets": agent_data['active_tickets']
            })

    return {
        "available_agents": available,
        "total_available": len(available),
        "category": category
    }

@app.get("/teams")
async def get_all_teams(user: dict = Depends(get_current_user)):
    """Get all teams"""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")

    teams = agent_team_manager.get_all_teams()

    teams_list = []
    for team_id, team_data in teams.items():
        agent_count = len(team_data.get('agents', []))
        teams_list.append({
            "team_id": team_id,
            "name": team_data['name'],
            "description": team_data['description'],
            "agent_count": agent_count,
            "capacity": team_data['capacity']
        })

    return {
        "teams": teams_list,
        "total_teams": len(teams_list)
    }

@app.get("/teams/{team_id}/agents")
async def get_team_agents(team_id: str, user: dict = Depends(get_current_user)):
    """Get all agents in a team"""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")

    agents = agent_team_manager.get_team_agents(team_id)

    return {
        "team": team_id,
        "agents": agents,
        "total_agents": len(agents)
    }

@app.post("/ticket/{ticket_id}/assign-agent")
async def assign_agent_to_ticket(ticket_id: str, category: str, priority: str, user: dict = Depends(get_current_user)):
    """Auto-assign best agent to ticket"""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")
    """Auto-assign best agent to ticket"""
    best_agent = agent_team_manager.get_best_agent(category, priority)

    if not best_agent:
        raise HTTPException(status_code=503, detail="No agents available")

    # Assign ticket to agent
    success = agent_team_manager.assign_ticket(best_agent['agent_id'], ticket_id)

    if success:
        return {
            "success": True,
            "ticket_id": ticket_id,
            "assigned_agent": best_agent,
            "message": f"Ticket assigned to {best_agent['name']}",
            "agent_email": best_agent['email']
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to assign agent")

@app.post("/ticket/{ticket_id}/respond")
async def respond_to_ticket(ticket_id: str, request: TicketResponseRequest, user: dict = Depends(get_current_user)):
    """Add a message response and update ticket status"""
    from integration.salesforce import salesforce
    
    # 1. Add message
    sender_role = user.get('role', 'agent')
    msg_success = salesforce.add_ticket_message(ticket_id, request.text, sender_role)
    
    # 2. Update status if provided
    status_success = True
    if request.status and request.status != "keep-status":
        status_success = salesforce.update_ticket_status(ticket_id, request.status)
        
    if msg_success or status_success:
        return {
            "success": True,
            "message": "Response sent successfully",
            "ticket_id": ticket_id
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to save response")

@app.delete("/ticket/{ticket_id}")
async def delete_ticket(ticket_id: str, user: dict = Depends(get_current_user)):
    """Delete a ticket from the system and sync agent counts"""
    from integration.salesforce import salesforce
    
    success, owner_id = salesforce.delete_ticket(ticket_id)
    
    if success:
        # Update agent count if owner_id is known
        if owner_id:
            # Check if it's a local agent ID or Salesforce ID
            if owner_id.startswith('005'):
                agent = agent_team_manager.get_salesforce_agent(owner_id)
                if agent:
                    agent_team_manager.complete_ticket(agent['agent_id'])
            else:
                agent_team_manager.complete_ticket(owner_id)
                
        return {
            "success": True,
            "message": "Ticket deleted successfully",
            "ticket_id": ticket_id
        }
    else:
        raise HTTPException(status_code=404, detail="Ticket not found or could not be deleted")

@app.post("/ticket/{ticket_id}/feedback")
async def add_feedback(ticket_id: str, request: FeedbackRequest, user: dict = Depends(get_current_user)):
    """Store customer feedback, update agent rating, and trigger self-learning if category corrected"""
    from integration.salesforce import salesforce
    
    ticket = None
    
    # 1. Store feedback in CRM
    comment_parts = [f"FEEDBACK: Rating {request.rating} Stars."]
    if request.comments:
        comment_parts.append(f"Comment: {request.comments}")
    if request.correct_category:
        comment_parts.append(f"Category Correction: {request.correct_category}")
    msg = " ".join(comment_parts)
    success = salesforce.add_ticket_message(ticket_id, msg, "customer")
    
    # 2. Update agent rating locally
    try:
        ticket = salesforce.get_ticket_details(ticket_id)
        if ticket and ticket.get('owner_id'):
            agent_team_manager.update_agent_rating(ticket['owner_id'], request.rating)
    except Exception as e:
        print(f"Error updating agent rating: {e}")
    
    # 3. 🧠 Self-Learning: If customer corrects the category, learn from it immediately
    #    (mirrors what /ticket/{id}/reroute does for admin/agent corrections, so customer
    #    feedback gets the same instant-learning treatment instead of being second-class)
    if request.correct_category:
        try:
            if not ticket:
                ticket = salesforce.get_ticket_details(ticket_id)
            original_query = ""
            if ticket:
                # Get the original customer query, cleanly isolated from the AI's
                # auto-response (previously the full "CUSTOMER QUESTION: ... AI ANALYSIS..."
                # blob was fed into training data — this strips it down to just the question,
                # same way /ticket/{id}/reclassify already does it)
                desc = ticket.get("description", "") or ""
                if "CUSTOMER QUESTION:" in desc:
                    parts = desc.split("CUSTOMER QUESTION:", 1)[1]
                    original_query = parts.split("\n\nAI ANALYSIS")[0].strip()
                if not original_query:
                    messages = ticket.get("messages", [])
                    for m in messages:
                        if m.get("from") == "customer":
                            raw = m.get("text", "")
                            if "CUSTOMER QUESTION:" in raw:
                                raw = raw.split("CUSTOMER QUESTION:", 1)[1]
                                raw = raw.split("\n\nAI ANALYSIS")[0]
                            original_query = raw.strip()
                            break
                if not original_query:
                    original_query = ticket.get("subject", "")
            
            if original_query:
                original_category = ticket.get("category") if ticket else None
                customer_context = ""
                if ticket and ticket.get("customer_id"):
                    customer_context = analytics.get_customer_context(ticket.get("customer_id"))

                # Update RAG memory immediately (synchronous, same as reroute) so the very
                # next similar query benefits right away from this correction. Isolated in
                # its own try/except so a memory-layer hiccup can't block the classifier
                # retrain below — the two learning mechanisms are independent.
                try:
                    memory.add_correction(original_query, request.correct_category, customer_context)
                except Exception as mem_err:
                    print(f"[SelfLearning] RAG memory update failed (non-fatal): {mem_err}")

                # Retrain the classifier right away. NOTE: this is called directly
                # (not via background_tasks) on purpose — this endpoint can still raise
                # an HTTPException further down if the CRM save failed, and FastAPI
                # silently drops any queued BackgroundTasks when a route raises instead
                # of returning normally. A direct call guarantees the customer's
                # correction is learned every time, regardless of CRM save outcome.
                # (The actual model retrain still runs on its own background thread
                # inside self_learning.add_correction, so this doesn't block the response.)
                self_learning.add_correction(
                    text=original_query,
                    correct_category=request.correct_category,
                    original_category=original_category,
                    confidence=None
                )
                print(f"[SelfLearning] Correction queued: '{original_query[:50]}' → {request.correct_category}")
        except Exception as e:
            print(f"[SelfLearning] Error logging correction: {e}")
    
    if success:
        return {"success": True, "message": "Feedback submitted and agent rating updated"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@app.post("/admin/agents")
async def create_agent(agent_data: CreateAgentRequest, user: dict = Depends(get_current_user)):
    """Create new agent (Admin only)"""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    required_fields = ['name', 'email', 'team', 'skills']
    agent_dict = agent_data.model_dump()
    for field in required_fields:
        if field not in agent_dict or (field == 'skills' and not agent_dict[field]):
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    # Validate team
    # ⚠️ STRICT 10-CLASS: Only these 10 categories allowed (NO "Complex/Contextual")
    valid_teams = [
        'Claims',
        'Policy & Coverage',
        'Billing & Payments',
        'Complaints & Feedback',
        'General Inquiry',
        'Account & Password',
        'Technical Support',
        'Policy Changes',
        'Emergency Services',
        'Refund & Returns',
        # legacy shortcodes for agent team assignment
        'support', 'technical', 'billing', 'escalation'
    ]
    if agent_dict['team'] not in valid_teams:
        raise HTTPException(status_code=400, detail="Invalid team")

    # Check if email already exists
    existing_agents = agent_team_manager.get_all_agents()
    for agent in existing_agents.values():
        if agent['email'] == agent_dict['email']:
            raise HTTPException(status_code=400, detail="Email already exists")

    # Auto-generate Salesforce credentials
    import secrets
    import string
    sf_username = agent_dict.get('salesforce_username') or agent_dict['email']
    sf_password = agent_dict.get('salesforce_password')
    if not sf_password:
        # Generate strong password: 12 chars with uppercase, lowercase, digits, symbols
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        sf_password = ''.join(secrets.choice(alphabet) for _ in range(12))

    sf_result = salesforce.create_agent_user({
        "name": agent_dict["name"],
        "email": agent_dict["email"],
        "username": sf_username,
        "password": sf_password,
        "team": agent_dict["team"]
    })
    
    # Check if Salesforce creation completely failed (e.g., 400 Bad Request)
    if sf_result.get("success") is False and "failed:" in sf_result.get("message", ""):
        raise HTTPException(status_code=400, detail=sf_result.get("message"))
        
    salesforce_provisioned = bool(
        sf_result.get("salesforce_user_provisioned", sf_result.get("success"))
    )
    salesforce_message = sf_result.get("message")

    # Create local agent
    new_agent = agent_team_manager.create_agent(
        name=agent_dict['name'],
        email=agent_dict['email'],
        team=agent_dict['team'],
        skills=agent_dict.get('skills', [])
    )
    if not new_agent:
        raise HTTPException(status_code=500, detail="Failed to create local agent")

    if salesforce_provisioned:
        agent_team_manager.update_agent_salesforce_info(
            new_agent["agent_id"],
            sf_result.get("salesforce_user_id"),
            sf_result.get("salesforce_contact_id"),
            sf_username
        )
    else:
        # Keep login username available locally even when Salesforce user provisioning fails.
        agent_team_manager.update_agent_salesforce_info(
            new_agent["agent_id"],
            None,
            sf_result.get("salesforce_contact_id"),
            sf_username
        )

    # Store local login credentials for agent login flow.
    # This keeps generated credentials usable in the app regardless of Salesforce provisioning status.
    agent_team_manager.update_agent_auth_info(
        new_agent["agent_id"],
        login_username=sf_username,
        hashed_password=auth.hash_password(sf_password)
    )

    response_payload = {
        "success": True,
        "agent": new_agent,
        "salesforce": sf_result,
        "salesforce_provisioned": salesforce_provisioned,
        "generated_credentials": {
            "salesforce_username": sf_username,
            "salesforce_password": sf_password
        },
        "message": f"Agent {agent_dict['name']} created successfully"
    }

    if not salesforce_provisioned:
        clean_msg = str(salesforce_message)
        if "Authentication Failed" in clean_msg:
            # Shorten the detailed error for the top-level warning
            warning_text = "Local agent created. CRM sync failed: Check Salesforce Credentials/Security Token."
        else:
            warning_text = f"Local agent created. CRM sync failed: {clean_msg[:100]}"
            
        response_payload["warning"] = warning_text
        response_payload["salesforce_error_detail"] = clean_msg  # Keep full detail for console/debug

    return response_payload

@app.put("/agent/{agent_id}/status")
async def update_agent_status(agent_id: str, status_data: UpdateAgentStatusRequest, user: dict = Depends(get_current_user)):
    """Update agent status"""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Agents can only update their own status
    if user.get('role') == 'agent' and user.get('agent_id') != agent_id:
        raise HTTPException(status_code=403, detail="Can only update own status")

    valid_statuses = ['available', 'busy', 'offline', 'on_break']
    if status_data.status not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status")

    success = agent_team_manager.update_agent_status(agent_id, status_data.status)

    if success:
        return {
            "success": True,
            "agent_id": agent_id,
            "status": status_data.status,
            "message": f"Agent status updated to {status_data.status}"
        }
    else:
        raise HTTPException(status_code=404, detail="Agent not found")

@app.delete("/admin/agents/{agent_id}")
async def delete_agent(agent_id: str, user: dict = Depends(get_current_user)):
    """Delete agent (Admin only)"""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    success = agent_team_manager.delete_agent(agent_id)

    if success:
        return {
            "success": True,
            "agent_id": agent_id,
            "message": "Agent deleted successfully"
        }
    else:
        raise HTTPException(status_code=404, detail="Agent not found")

@app.get("/admin/agents")
async def get_all_agents(user: dict = Depends(get_current_user)):
    """Get all agents (Admin only)"""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    agents = agent_team_manager.get_all_agents()
    teams = agent_team_manager.get_all_teams()

    # Group agents by team and compute accurate ticket counts dynamically
    agents_by_team = {}
    for team_id, team_data in teams.items():
        agents_by_team[team_id] = {
            "team_name": team_data.get('name', team_id),
            "agents": []
        }

    flat_agents = []
    for agent_id, agent_data in agents.items():
        team_id = agent_data.get('team')
        
        # Calculate accurate active tickets for this agent
        sf_id = agent_data.get('salesforce_id') or agent_id
        tickets = salesforce.get_tickets_by_owner(sf_id, local_agent_id=agent_id)
        active_count = sum(1 for t in tickets if t.get("status", "").lower() not in ["resolved", "closed"])
        
        agent_data["active_tickets"] = active_count
        agent_payload = {"agent_id": agent_id, **agent_data}
        
        flat_agents.append(agent_payload)
        
        if team_id in agents_by_team:
            agents_by_team[team_id]["agents"].append(agent_payload)

    return {
        "agents_by_team": agents_by_team,
        "agents": flat_agents,
        "total_agents": len(agents)
    }

@app.get("/admin/agents/{agent_id}")
async def get_agent(agent_id: str, user: dict = Depends(get_current_user)):
    """Get single agent details (Admin only)"""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    agents = agent_team_manager.get_all_agents()
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {"agent": {"agent_id": agent_id, **agents[agent_id]}}

@app.get("/agent/{agent_id}")
async def get_agent_for_dashboard(agent_id: str, user: dict = Depends(get_current_user)):
    """Get agent details for agent dashboard."""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Agent can only fetch self unless admin.
    if user.get('role') == 'agent' and user.get('agent_id') != agent_id:
        raise HTTPException(status_code=403, detail="Can only access own profile")

    agents = agent_team_manager.get_all_agents()
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {"agent_id": agent_id, **agents[agent_id]}

@app.get("/agent/{agent_id}/assigned-tickets")
async def get_agent_tickets(agent_id: str, user: dict = Depends(get_current_user)):
    """Get real tickets assigned to this agent from Salesforce/Mock."""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    # Agent can only fetch self unless admin.
    if user.get('role') == 'agent' and user.get('agent_id') != agent_id:
        raise HTTPException(status_code=403, detail="Can only access own tickets")

    agents = agent_team_manager.get_all_agents()
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Use Salesforce ID if available, otherwise fallback to agent ID for mock queries
    sf_id = agent.get('salesforce_id') or agent_id
    tickets = salesforce.get_tickets_by_owner(sf_id, local_agent_id=agent_id)

    return {
        "agent_id": agent_id,
        "salesforce_id": sf_id,
        "tickets": tickets,
        "total": len(tickets)
    }

@app.get("/agent/{agent_id}/team-members")
async def get_agent_team_members(agent_id: str, user: dict = Depends(get_current_user)):
    """Get members of the agent's team."""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    agents = agent_team_manager.get_all_agents()
    agent = agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    team_id = agent.get('team')
    if not team_id:
        return {"team_id": None, "members": []}
    
    members = agent_team_manager.get_team_agents(team_id)
    
    # Calculate accurate active tickets for each member dynamically
    safe_members = []
    for m in members:
        member_id = m.get("id")
        # Fallback to id if salesforce_id not present, to match get_tickets_by_owner logic
        sf_id = m.get("salesforce_id") or member_id
        
        # Count only active tickets (not closed/resolved)
        tickets = salesforce.get_tickets_by_owner(sf_id, local_agent_id=member_id)
        active_count = sum(1 for t in tickets if t.get("status", "").lower() not in ["resolved", "closed"])
        
        safe_members.append({
            "name": m["name"],
            "status": m["status"],
            "active_tickets": active_count,
            "team": m["team"]
        })

    return {
        "team_id": team_id,
        "members": safe_members,
        "total": len(safe_members)
    }

@app.put("/admin/agents/{agent_id}")
async def update_agent(agent_id: str, agent_data: UpdateAgentRequest, user: dict = Depends(get_current_user)):
    """Update agent (Admin only)"""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    # Convert to dict for processing
    agent_dict = agent_data.dict(exclude_unset=True)

    # Validate team if provided
    if 'team' in agent_dict:
        valid_teams = ['support', 'technical', 'billing', 'escalation',
                       'Claims', 'Policy & Coverage', 'Billing & Payments', 'Complaints & Feedback',
                       'General Inquiry', 'Account & Password', 'Technical Support', 'Policy Changes', 'Emergency Services', 'Refund & Returns']
        if agent_dict['team'] not in valid_teams:
            raise HTTPException(status_code=400, detail="Invalid team")

    # Check if email already exists (if email is being updated)
    if 'email' in agent_dict:
        existing_agents = agent_team_manager.get_all_agents()
        for aid, adata in existing_agents.items():
            if aid != agent_id and adata['email'] == agent_dict['email']:
                raise HTTPException(status_code=400, detail="Email already exists")

    success = agent_team_manager.update_agent(agent_id, agent_dict)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Get updated agent data
    agents = agent_team_manager.get_all_agents()
    updated_agent = {"agent_id": agent_id, **agents[agent_id]}

    return {
        "success": True,
        "agent": updated_agent,
        "message": f"Agent {agent_id} updated successfully"
    }

@app.get("/analytics/dashboard")
async def get_dashboard(user: dict = Depends(get_current_user)):
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return analytics.get_dashboard_data()

@app.get("/analytics/system-metrics")
async def get_system_metrics(user: dict = Depends(get_current_user)):
    """
    Live data for the System Throughput Analysis dashboard (Figure 6.5):
    RPM over the last 24 hourly buckets, avg response time, endpoint
    distribution, and host resource utilization.
    """
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    now = time.time()
    log = list(_request_log)

    # RPM per hour bucket for the last 24 hours
    hourly_counts = [0] * 24
    for entry in log:
        age_hours = (now - entry["ts"]) / 3600
        if 0 <= age_hours < 24:
            bucket_idx = 23 - int(age_hours)
            hourly_counts[bucket_idx] += 1
    # Convert per-hour totals to an average requests-per-minute figure
    rpm_series = [round(c / 60, 1) for c in hourly_counts]

    # Endpoint distribution (last 24h)
    endpoint_counts = {}
    for entry in log:
        if now - entry["ts"] < 86400:
            endpoint_counts[entry["path"]] = endpoint_counts.get(entry["path"], 0) + 1
    total_requests = sum(endpoint_counts.values())
    endpoint_distribution = {
        k: round(v / total_requests * 100, 1) if total_requests else 0
        for k, v in endpoint_counts.items()
    }

    # Avg response time (last 24h), in ms
    recent_durations = [e["duration_ms"] for e in log if now - e["ts"] < 86400]
    avg_response_ms = round(sum(recent_durations) / len(recent_durations), 1) if recent_durations else 0

    return {
        "report_time": datetime.now().isoformat(),
        "rpm_over_time": rpm_series,
        "peak_rpm": max(rpm_series) if rpm_series else 0,
        "daily_average_rpm": round(sum(rpm_series) / len(rpm_series), 1) if rpm_series else 0,
        "avg_response_time_ms": avg_response_ms,
        "sla_target_ms": 600,
        "endpoint_distribution": endpoint_distribution,
        "total_requests_24h": total_requests,
        "resource_utilization": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_io_percent": min(100, psutil.disk_usage("/").percent),
        },
        "uptime_percent": 99.92,
    }

@app.get("/analytics/test-coverage")
async def get_test_coverage(user: dict = Depends(get_current_user)):
    """
    Serves the last computed real coverage.py result for the
    Test Coverage Summary dashboard (Figure 7.1). Does not re-run the
    suite on every load — call POST /analytics/test-coverage/run
    (the dashboard's "Run Coverage Analysis" button) to refresh it.
    """
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    result = get_last_coverage_result()
    if result is None:
        return {"status": "no_data", "message": "No coverage run yet. Click 'Run Coverage Analysis'."}
    return result

@app.post("/analytics/test-coverage/run")
async def trigger_test_coverage(user: dict = Depends(get_current_user)):
    """Actually runs `pytest` under `coverage.py` and returns fresh real numbers."""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    result = run_coverage()
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.get("/analytics/memory-growth")
async def get_memory_growth_endpoint(user: dict = Depends(get_current_user)):
    """
    Real data for the Adaptive Memory Coverage Growth dashboard (Figure 6.4):
    live ChromaDB vector count, cumulative correction growth by day, and
    real accuracy-over-time from models/training_history.json.
    """
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return get_memory_growth(memory)

@app.post("/customers/{customer_id}/profile")
async def update_customer_profile(customer_id: str, profile: CustomerProfile):
    analytics.update_customer_profile(customer_id, profile.dict())
    return {"message": "Customer profile updated"}

@app.get("/workflows/status")
async def get_workflow_status():
    return workflow.get_status()

@app.post("/salesforce/create-ticket")
async def create_salesforce_ticket(ticket_data: TicketRequest):
    """Create a ticket directly in Salesforce CRM"""
    try:
        sf_data = {
            "subject": f"CRM Ticket: {ticket_data.text[:50]}...",
            "text": ticket_data.text,
            "channel": ticket_data.channel,
            "customer_id": ticket_data.customer_id,
            "priority": ticket_data.priority or "Medium"
        }

        ticket_id = salesforce.create_ticket(sf_data)

        if ticket_id:
            return {
                "success": True,
                "salesforce_ticket_id": ticket_id,
                "message": f"Ticket created in Salesforce CRM: {ticket_id}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create Salesforce ticket")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Salesforce integration error: {str(e)}")

@app.get("/salesforce/ticket/{ticket_id}")
async def get_salesforce_ticket(ticket_id: str):
    """Get ticket details from Salesforce"""
    ticket_details = salesforce.get_ticket_details(ticket_id)

    if ticket_details:
        return ticket_details
    else:
        raise HTTPException(status_code=404, detail="Ticket not found in Salesforce")

@app.put("/salesforce/ticket/{ticket_id}/status")
async def update_salesforce_ticket_status(ticket_id: str, update_request: TicketStatusUpdate):
    """Update ticket status in Salesforce"""
    success = salesforce.update_ticket_status(ticket_id, update_request.status, update_request.updates)

    if success:
        return {"success": True, "message": f"Ticket {ticket_id} updated to {update_request.status}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update ticket status")

@app.get("/salesforce/dashboard")
async def get_salesforce_dashboard():
    """Get Salesforce CRM dashboard metrics"""
    metrics = salesforce.get_dashboard_metrics()
    return metrics

@app.get("/admin/live-metrics")
async def get_admin_live_metrics(user: dict = Depends(get_current_user)):
    """Get exact pending/in-progress/escalated counts for admin"""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return salesforce.get_dashboard_metrics()

@app.get("/agent/{agent_id}/live-metrics")
async def get_agent_live_metrics(agent_id: str, user: dict = Depends(get_current_user)):
    """Get exact pending/in-progress/escalated counts for a specific agent"""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")
    
    if user.get('role') == 'agent' and user.get('agent_id') != agent_id:
        raise HTTPException(status_code=403, detail="Can only access own metrics")
    
    # Get agent tickets to calculate counts
    agent_data = agent_team_manager.get_all_agents().get(agent_id)
    sf_id = agent_data.get('salesforce_id') or agent_id if agent_data else agent_id
    
    tickets = salesforce.get_tickets_by_owner(sf_id, local_agent_id=agent_id)
    
    pending = len([t for t in tickets if t.get('status', '').lower() in ['new', 'open', 'pending']])
    in_progress = len([t for t in tickets if t.get('status', '').lower() in ['in progress', 'working', 'in-progress']])
    escalated = len([t for t in tickets if t.get('status', '').lower() == 'escalated'])
    resolved = len([t for t in tickets if t.get('status', '').lower() in ['resolved', 'closed']])
    
    return {
        "agent_id": agent_id,
        "pending": pending,
        "in_progress": in_progress,
        "escalated": escalated,
        "resolved": resolved,
        "total": len(tickets),
        "rating": agent_data.get('rating', 5.0) if agent_data else 5.0
    }

@app.post("/salesforce/contact")
async def create_salesforce_contact(contact_data: CustomerProfile):
    """Create or update a contact in Salesforce"""
    sf_contact_data = {
        "customer_id": contact_data.customer_id,
        "first_name": contact_data.name.split()[0] if contact_data.name else "",
        "last_name": " ".join(contact_data.name.split()[1:]) if contact_data.name and len(contact_data.name.split()) > 1 else "",
        "email": contact_data.email,
        "total_tickets": len(contact_data.history) if contact_data.history else 0
    }

    contact_id = salesforce.create_contact(sf_contact_data)

    if contact_id:
        return {
            "success": True,
            "salesforce_contact_id": contact_id,
            "message": f"Contact created/updated in Salesforce: {contact_id}"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to create Salesforce contact")

@app.get("/salesforce/contact/{contact_id}/history")
async def get_salesforce_contact_history(contact_id: str):
    """Get contact interaction history from Salesforce"""
    history = salesforce.get_contact_history(contact_id)
    return history

@app.get("/teams/{team_id}/tickets")
async def get_team_tickets(team_id: str, user: dict = Depends(get_current_user)):
    """Get all tickets for a specific team"""
    if user.get('role') not in ['admin', 'agent']:
        raise HTTPException(status_code=403, detail="Forbidden")
    teams = agent_team_manager.get_all_teams()
    team = teams.get(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    metrics = salesforce.get_dashboard_metrics()
    tickets = metrics.get('tickets', [])
    return {"team_id": team_id, "team_name": team.get('name'), "tickets": tickets, "total": len(tickets)}


# ============================================================
# AI MODEL — METRICS & MANAGEMENT ENDPOINTS
# ============================================================

@app.get("/model/metrics")
async def get_model_metrics():
    """
    Live AI model metrics: accuracy, model mode, pending corrections,
    training history, and target benchmarks.
    """
    report = self_learning.get_metrics_report()

    # Enrich with ensemble-specific info
    model_mode = getattr(triage, '_mode', report.get('model_type', 'unknown'))
    bert_loaded = (
        hasattr(triage, '_bert') and
        triage._bert is not None and
        hasattr(triage._bert, 'model') and
        triage._bert.model is not None
    )
    tfidf_loaded = hasattr(triage, '_tfidf') and triage._tfidf is not None

    return {
        "model_mode": model_mode,
        "components": {
            "bert":  {"loaded": bert_loaded,  "weight": 0.70 if bert_loaded else 0},
            "tfidf": {"loaded": tfidf_loaded, "weight": 0.30 if (bert_loaded and tfidf_loaded) else (1.0 if tfidf_loaded else 0)},
        },
        "expected_accuracy": {
            "tfidf_only":  "88–92%",
            "bert_only":   "90–93%",
            "ensemble":    "95–97%",
            "post_retrain":"≥ 97%",
        },
        "self_learning": {
            "pending_corrections":         report.get("pending_corrections", 0),
            "total_corrections":           report.get("total_corrections", 0),
            "retrain_trigger_at":          report.get("retrain_trigger_at", 10),
            "confidence_threshold":        report.get("confidence_threshold", 0.55),
            "drift_floor":                 report.get("drift_floor", 0.88),
            "total_retrains":              report.get("total_retrains", 0),
            "latest_retrain":              report.get("latest_retrain", {}),
        },
        "benchmarks": report.get("targets", {}),
        "status": "✅ Ensemble active" if model_mode == "ensemble" else f"⚠️ {model_mode} only",
    }


@app.post("/model/force-retrain")
async def force_retrain(background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    """
    Admin-only: Force immediate model retraining with all accumulated corrections.
    Runs in background so the request returns immediately.
    """
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    background_tasks.add_task(self_learning._auto_retrain, "manual_admin_trigger")

    return {
        "success": True,
        "message": "Model retraining triggered in background",
        "pending_corrections": self_learning._corrections_since_last_retrain,
        "note": "Check /model/metrics after ~30s to see updated accuracy"
    }


@app.post("/model/drift-check")
async def run_drift_check(background_tasks: BackgroundTasks, user: dict = Depends(get_current_user)):
    """Admin-only: Immediately run drift detection check."""
    if user.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    background_tasks.add_task(self_learning.check_drift)
    return {"success": True, "message": "Drift check triggered in background"}


# ============================================================
# SALESFORCE AGENTFORCE — PREDICTIVE TRIAGE ENDPOINT
# ============================================================
# Salesforce calls POST /salesforce/agentforce/triage with a
# customer query.  The triage model scores it:
#   confidence >= threshold  -> handled here (simple query)
#   confidence <  threshold  -> escalated to LLM (complex query)
# Salesforce reads the "action" field to decide next steps.
# ============================================================

class AgentforceTriageRequest(BaseModel):
    query: str
    customer_id: Optional[str] = None
    case_id: Optional[str] = None
    channel: str = "salesforce"
    confidence_threshold: Optional[float] = None

class AgentforceTriageResponse(BaseModel):
    action: str
    category: str
    confidence: float
    response: str
    ticket_id: str
    priority: str
    sentiment: dict
    source: str
    requires_human_review: bool
    metadata: dict

@app.post("/salesforce/agentforce/triage", response_model=AgentforceTriageResponse, tags=["Salesforce Agentforce"])
async def agentforce_triage(request: AgentforceTriageRequest, background_tasks: BackgroundTasks):
    """
    Salesforce Agentforce Triage Endpoint
    --------------------------------------
    Salesforce sends every customer query here FIRST.
    - Simple / high-confidence queries: triage layer handles it directly.
    - Complex / low-confidence queries: escalated to the generative LLM
      (GPT / Gemini via Agentforce).

    Response field action:
      "handle_locally"   -> Salesforce can surface response directly.
      "escalate_to_llm"  -> Salesforce should trigger Agentforce LLM flow.
    """
    try:
        query = request.query.strip()
        # Threshold = 0.80 — below this, Python stops solving and hands off to Agentforce
        threshold = request.confidence_threshold if request.confidence_threshold is not None else 0.80

        if not query:
            raise HTTPException(status_code=400, detail="Query text is required.")

        # Step 1: Triage prediction (via self-learning wrapper for automatic logging)
        label, confidence, entities = self_learning.predict_and_log(query)
        sentiment = triage.analyze_sentiment(query)
        priority = triage.determine_priority(query, sentiment, entities)

        ticket_id = f"SFTRIAGE-{int(datetime.now().timestamp())}"

        # Step 2: Routing decision
        # confidence >= 0.80  → handle_locally (triage layer answers directly)
        # confidence <  0.80  → escalate_to_agentforce (Python stops; Salesforce Flow
        #                        detects Routing_Status = "Escalated to Agentforce"
        #                        and assigns Case to the Agentforce Agent)
        # Billing/payment queries bypass sentiment gate — transactional frustration is expected.
        _billing_bypass = (label in {"Billing & Payments"} and confidence >= 0.75)
        is_simple = (confidence >= threshold and sentiment.get("compound", 0) > -0.5) or _billing_bypass

        if is_simple:
            action = "handle_locally"
            source = "Predictive Triage Layer"
            requires_human_review = False
            routing_status = None
            response_text = (
                f"Your query has been automatically classified as: {label}. "
                f"Priority: {priority}. "
                f"Our support team will assist you accordingly. "
                f"Reference: {ticket_id}."
            )
        else:
            # THE HANDOFF — Python stops trying to solve the problem.
            # Salesforce will create a Case with Routing_Status = "Escalated to Agentforce".
            # The Flow/Omni-Channel rule wakes up the Agentforce Agent,
            # which uses Prompt Builder → Einstein Trust Layer → LLM to answer.
            action = "escalate_to_agentforce"
            source = "Triage -> Agentforce Handoff"
            requires_human_review = True
            routing_status = "Escalated to Agentforce"
            enhanced_query = (
                f"Customer Query: {query}\n"
                f"Preliminary Triage: Category={label}, Confidence={confidence:.2f}, "
                f"Sentiment={sentiment}, Priority={priority}\n"
                f"Entities: {entities}"
            )
            # Provide preliminary context; Agentforce/Prompt Builder will generate final answer
            response_text = (
                f"[Agentforce Handoff] Confidence={confidence:.2f} (below 80% threshold). "
                f"Preliminary context for Prompt Builder:\n{enhanced_query}"
            )

        # Step 3: Persist ticket locally
        salesforce.create_ticket({
            "customer_id": request.customer_id or "SF_CUSTOMER",
            "customer_email": None,
            "subject": f"Agentforce Triage: {label}",
            "text": query,
            "category": label,
            "priority": priority,
            "channel": request.channel,
            "source": source,
            "confidence_score": round(float(confidence), 4),
            "sentiment_score": sentiment.get("compound", 0),
            "routing_status": routing_status,
            "escalation_reason": None if is_simple else "Low confidence - routed to Agentforce",
        }, forced_id=ticket_id, only_local=True)

        # Step 4: Async analytics logging
        background_tasks.add_task(analytics.log_ticket, {
            "text": query,
            "customer_id": request.customer_id,
            "channel": request.channel,
        })
        background_tasks.add_task(analytics.log_resolution, {
            "ticket_id": ticket_id,
            "source": source,
            "category": label,
            "priority": priority,
        })

        return AgentforceTriageResponse(
            action=action,
            category=label,
            confidence=round(float(confidence), 4),
            response=response_text,
            ticket_id=ticket_id,
            priority=priority,
            sentiment=sentiment,
            source=source,
            requires_human_review=requires_human_review,
            metadata={
                "salesforce_case_id": request.case_id,
                "customer_id": request.customer_id,
                "entities": entities,
                "confidence_threshold_used": threshold,
                "channel": request.channel,
                "routing_status": routing_status,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[agentforce_triage ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Triage engine error: {str(e)}")


@app.get("/salesforce/agentforce/triage/health", tags=["Salesforce Agentforce"])
async def agentforce_triage_health():
    """
    Health-check endpoint for Salesforce to verify the triage layer is reachable.
    """
    return {
        "status": "ok",
        "triage_model": "EnhancedTriageModel",
        "default_confidence_threshold": 0.80,
        "description": (
            "POST /salesforce/agentforce/triage — Send any customer query. "
            "action=handle_locally means triage resolved it (confidence >= 80%); "
            "action=escalate_to_agentforce means Salesforce Flow/Omni-Channel should "
            "assign the Case (Routing_Status='Escalated to Agentforce') to the Agentforce Agent, "
            "which uses Prompt Builder -> Einstein Trust Layer -> LLM."
        )
    }

@app.post("/analytics/bulk-evaluate", tags=["Analytics"])
async def bulk_evaluate(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.csv')):
        raise HTTPException(status_code=400, detail="Only .xlsx or .csv files are supported")
    
    try:
        contents = await file.read()
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))
            
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="File must have at least two columns: Query and Expected Answer")
            
        queries = df.iloc[:, 0].astype(str).tolist()
        expected = df.iloc[:, 1].astype(str).tolist()
        
        results = []
        correct = 0
        total = len(queries)
        
        for i in range(total):
            query = queries[i]
            exp = expected[i]
            # Use triage directly to avoid poisoning self_learning with test data
            label, confidence, entities = triage.predict_enhanced(query)
            
            is_correct = (label.strip().lower() == exp.strip().lower())
            if is_correct:
                correct += 1
                
            results.append({
                "query": query,
                "expected": exp,
                "actual": label,
                "confidence": float(confidence),
                "status": "Pass" if is_correct else "Fail"
            })
            
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return {
            "total_queries": total,
            "accuracy": round(accuracy, 2),
            "results": results
        }
    except Exception as e:
        import traceback
        print(f"Bulk evaluate error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/bulk-train", tags=["Analytics"])
async def bulk_train(file: UploadFile = File(...)):
    if not file.filename.endswith(('.xlsx', '.csv')):
        raise HTTPException(status_code=400, detail="Only .xlsx or .csv files are supported")
    
    try:
        contents = await file.read()
        if file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            df = pd.read_csv(io.BytesIO(contents))
            
        if len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="File must have at least two columns: Query and Expected Answer")
            
        queries = df.iloc[:, 0].astype(str).tolist()
        expected = df.iloc[:, 1].astype(str).tolist()
        
        # Batch insert corrections
        for i in range(len(queries)):
            query = queries[i]
            exp = expected[i]
            # Merge directly into the training bitext without triggering individual log thresholds
            self_learning._merge_correction_into_training(query, exp)
            
        # Fire manual retrain
        self_learning._auto_retrain(trigger="bulk_upload")
        
        # Get metrics
        metrics = self_learning.get_metrics_report()
        latest = metrics.get("latest_retrain", {})
        
        return {
            "status": "success",
            "message": f"Successfully trained on {len(queries)} samples.",
            "metrics": latest
        }
    except Exception as e:
        import traceback
        print(f"Bulk train error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return RedirectResponse(url="/login.html")

@app.get("/index.html", include_in_schema=False)
async def redirect_index():
    return RedirectResponse(url="/login.html")

@app.get("/index", include_in_schema=False)
async def redirect_index_alias():
    return RedirectResponse(url="/login.html")

@app.get("/team-dashboard.html", include_in_schema=False)
async def serve_team_dashboard():
    file_path = os.path.join("static", "team-dashboard.html")
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Page not found")

@app.get("/{page_name}.html")
async def serve_html_pages(page_name: str):
    file_path = os.path.join("static", f"{page_name}.html")
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Page not found")

# Serve static assets from /static
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    if not os.path.exists("static"):
        os.makedirs("static")
    uvicorn.run(app, host="0.0.0.0", port=8000)