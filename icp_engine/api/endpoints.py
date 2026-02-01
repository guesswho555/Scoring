"""
FastAPI Endpoints for ICP Scoring Engine
========================================
RESTful API for lead qualification and ICP scoring.

Base URL: http://localhost:8000

Endpoints:
- GET  /                          - API info
- GET  /api/health                - Health check
- POST /api/icp/score             - Score a single lead
- POST /api/icp/score/batch       - Score multiple leads
- POST /api/icp/score/quick       - Quick score (no LLM)
- POST /api/icp/explain           - Generate LLM explanation
- POST /api/icp/configure         - Create/update ICP config
- GET  /api/icp/configure         - List ICP configs
- GET  /api/icp/configure/{id}    - Get ICP config
- DELETE /api/icp/configure/{id}  - Delete ICP config
- GET  /api/stats                 - Get engine statistics
- POST /api/test                  - Test with sample data
"""

import os
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

from ..models.schemas import (
    LeadData,
    UserCompanyProfile,
    ICPScoreResult,
    ScoreRequest,
    BatchScoreRequest,
    BatchScoreResult,
)
from ..models.icp_config import ICPConfig, create_default_icp_config
from ..engine import ICPScoringEngine


# =============================================================================
# FastAPI App Initialization
# =============================================================================

app = FastAPI(
    title="ICP Scoring Engine API",
    description="""
## Lead Qualification & ICP Scoring System

This API provides intelligent lead scoring based on your Ideal Customer Profile (ICP).

### Features:
- **4-Stage Pipeline**: Filters → Signals → Scoring → LLM Analysis
- **Real-time Scoring**: Score leads in milliseconds
- **LLM Intelligence**: AI-powered analysis via OpenRouter
- **Batch Processing**: Score multiple leads efficiently

### Quick Start:
1. Use `/api/icp/score/quick` for fast scoring without LLM
2. Use `/api/icp/score` with `force_llm: true` for full AI analysis
3. Use `/api/icp/score/batch` for bulk processing
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - Allow all origins for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Storage & Engine Initialization
# =============================================================================

# In-memory storage (replace with database in production)
icp_configs: Dict[str, ICPConfig] = {}
engines: Dict[str, ICPScoringEngine] = {}

# Initialize default engine with API key from environment
def get_default_engine() -> ICPScoringEngine:
    api_key = os.getenv("OPENROUTER_API_KEY")
    return ICPScoringEngine(llm_api_key=api_key, llm_provider="openrouter")

default_engine = get_default_engine()


# =============================================================================
# Request/Response Models for Frontend
# =============================================================================

class QuickScoreRequest(BaseModel):
    """Simplified request for quick scoring"""
    company_name: str = Field(..., description="Company name")
    industry: Optional[str] = Field(None, description="Industry sector")
    description: Optional[str] = Field(None, description="Company description")
    employee_count: Optional[int] = Field(None, description="Number of employees")
    headquarters: Optional[str] = Field(None, description="Company location")
    technologies: Optional[List[str]] = Field(default=[], description="Tech stack")
    funding_total: Optional[float] = Field(None, description="Total funding raised")
    funding_round: Optional[str] = Field(None, description="Last funding round")
    website: Optional[str] = Field(None, description="Company website URL")

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "TechStartup Inc",
                "industry": "SaaS",
                "description": "B2B software company building AI analytics",
                "employee_count": 85,
                "headquarters": "San Francisco, CA",
                "technologies": ["AWS", "Python", "React"],
                "funding_total": 12000000,
                "funding_round": "Series A"
            }
        }


class QuickScoreResponse(BaseModel):
    """Simplified response for frontend"""
    company_name: str
    score: int
    grade: str
    status: str
    priority: str
    summary: str
    next_step: str
    signals: Dict[str, Any]
    breakdown: Dict[str, float]
    processing_time_ms: float


class FullScoreRequest(BaseModel):
    """Full scoring request with all options"""
    lead: QuickScoreRequest
    user_company: Optional[Dict[str, Any]] = Field(None, description="Your company profile")
    use_llm: bool = Field(False, description="Enable LLM analysis (slower but richer)")
    icp_id: Optional[str] = Field(None, description="ICP configuration ID")

    class Config:
        json_schema_extra = {
            "example": {
                "lead": {
                    "company_name": "TechStartup Inc",
                    "industry": "SaaS",
                    "description": "B2B SaaS company with Series A funding",
                    "employee_count": 85,
                    "technologies": ["AWS", "Python"]
                },
                "user_company": {
                    "name": "Acme Platform",
                    "core_offering": "API Integration",
                    "target_industries": ["SaaS", "FinTech"]
                },
                "use_llm": True
            }
        }


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API information and available endpoints"""
    return {
        "service": "ICP Scoring Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "Quick Score": "POST /api/icp/score/quick",
            "Full Score": "POST /api/icp/score",
            "Batch Score": "POST /api/icp/score/batch",
            "LLM Explain": "POST /api/icp/explain",
            "Test": "POST /api/test",
            "Health": "GET /api/health",
        }
    }


@app.get("/api/health", tags=["Info"])
async def health_check():
    """Health check endpoint for monitoring"""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    return {
        "status": "healthy",
        "service": "ICP Scoring Engine",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "llm_configured": bool(api_key and len(api_key) > 10),
    }


# =============================================================================
# Main Scoring Endpoints
# =============================================================================

@app.post("/api/icp/score/quick", response_model=QuickScoreResponse, tags=["Scoring"])
async def quick_score(request: QuickScoreRequest):
    """
    Quick score a lead without LLM (fast, ~50ms)

    Use this for:
    - Real-time scoring as user types
    - Bulk filtering before detailed analysis
    - When you don't need AI explanations
    """
    # Convert to LeadData
    lead = LeadData(
        company_name=request.company_name,
        industry=request.industry,
        description=request.description,
        employee_count=request.employee_count,
        headquarters=request.headquarters,
        technologies=request.technologies or [],
        url=request.website,
        funding_info={
            "total_raised": request.funding_total,
            "last_round": request.funding_round,
        } if request.funding_total or request.funding_round else None,
    )

    # Score without LLM
    result = default_engine.score_lead(lead, skip_llm=True)

    # Format response for frontend
    return QuickScoreResponse(
        company_name=result.company_name,
        score=result.scoring.overall_score,
        grade=result.scoring.grade.value,
        status=result.scoring.qualification_status.value,
        priority=result.summary.get("priority_rank", "P3"),
        summary=result.summary.get("one_liner", ""),
        next_step=result.summary.get("next_step", ""),
        signals={
            "positive": [{"tag": s.tag, "evidence": s.matched_text} for s in result.signals.positive[:5]],
            "negative": [{"tag": s.tag, "evidence": s.matched_text} for s in result.signals.negative[:3]],
            "net_score": result.signals.net_signal_score,
        },
        breakdown={
            "firmographic": result.scoring.breakdown.firmographic.score,
            "technographic": result.scoring.breakdown.technographic.score,
            "behavioral": result.scoring.breakdown.behavioral.score,
            "signal_boost": result.scoring.breakdown.signal_boost.score,
        },
        processing_time_ms=result.total_processing_time_ms,
    )


@app.post("/api/icp/score", tags=["Scoring"])
async def score_lead(request: FullScoreRequest):
    """
    Score a lead with optional LLM analysis

    Set `use_llm: true` for:
    - Detailed match explanations
    - AI-generated talking points
    - Risk identification
    - Relationship classification
    """
    # Get engine
    engine = _get_engine(request.icp_id)

    # Convert to LeadData
    lead = LeadData(
        company_name=request.lead.company_name,
        industry=request.lead.industry,
        description=request.lead.description,
        employee_count=request.lead.employee_count,
        headquarters=request.lead.headquarters,
        technologies=request.lead.technologies or [],
        url=request.lead.website,
        funding_info={
            "total_raised": request.lead.funding_total,
            "last_round": request.lead.funding_round,
        } if request.lead.funding_total or request.lead.funding_round else None,
    )

    # Convert user company if provided
    user_company = None
    if request.user_company:
        user_company = UserCompanyProfile(**request.user_company)

    # Score
    result = engine.score_lead(
        lead=lead,
        user_company=user_company,
        skip_llm=not request.use_llm,
        force_llm=request.use_llm,
    )

    # Build response
    response = {
        "company_name": result.company_name,
        "score": result.scoring.overall_score,
        "grade": result.scoring.grade.value,
        "status": result.scoring.qualification_status.value,
        "priority": result.summary.get("priority_rank", "P3"),
        "summary": result.summary,
        "scoring": {
            "overall": result.scoring.overall_score,
            "grade": result.scoring.grade.value,
            "breakdown": {
                "firmographic": {
                    "score": result.scoring.breakdown.firmographic.score,
                    "weighted": result.scoring.breakdown.firmographic.weighted,
                    "details": result.scoring.breakdown.firmographic.details,
                },
                "technographic": {
                    "score": result.scoring.breakdown.technographic.score,
                    "weighted": result.scoring.breakdown.technographic.weighted,
                    "details": result.scoring.breakdown.technographic.details,
                },
                "behavioral": {
                    "score": result.scoring.breakdown.behavioral.score,
                    "weighted": result.scoring.breakdown.behavioral.weighted,
                    "details": result.scoring.breakdown.behavioral.details,
                },
                "signal_boost": {
                    "score": result.scoring.breakdown.signal_boost.score,
                    "weighted": result.scoring.breakdown.signal_boost.weighted,
                    "details": result.scoring.breakdown.signal_boost.details,
                },
            },
            "data_completeness": result.scoring.data_completeness,
        },
        "signals": {
            "positive": [
                {"tag": s.tag, "category": s.category, "weight": s.weight, "evidence": s.matched_text}
                for s in result.signals.positive
            ],
            "negative": [
                {"tag": s.tag, "category": s.category, "weight": s.weight, "evidence": s.matched_text}
                for s in result.signals.negative
            ],
            "total_boost": result.signals.total_boost,
            "total_penalty": result.signals.total_penalty,
        },
        "filter_passed": result.filter_status.passed,
        "processing_time_ms": result.total_processing_time_ms,
    }

    # Add LLM analysis if available
    if result.intelligence:
        response["llm_analysis"] = {
            "relationship_type": result.intelligence.relationship_type.value,
            "confidence": result.intelligence.relationship_confidence,
            "explanation": result.intelligence.match_explanation,
            "talking_points": result.intelligence.talking_points,
            "risks": [
                {
                    "risk": r.risk,
                    "severity": r.severity.value,
                    "mitigation": r.mitigation,
                }
                for r in result.intelligence.risks
            ],
            "recommended_action": result.intelligence.recommended_action,
            "suggested_channel": result.intelligence.suggested_channel,
            "urgency": result.intelligence.urgency,
            "processing_time_ms": result.intelligence.processing_time_ms,
        }

    return response


@app.post("/api/icp/score/batch", tags=["Scoring"])
async def score_batch(
    leads: List[QuickScoreRequest] = Body(..., description="List of leads to score"),
    use_llm_for_top: int = Query(0, description="Run LLM for top N leads"),
    min_score: int = Query(0, description="Minimum score threshold"),
):
    """
    Score multiple leads at once

    - Fast parallel processing
    - Optional LLM for top N results only (cost optimization)
    - Results sorted by score
    """
    # Convert leads
    lead_objects = []
    for req in leads:
        lead_objects.append(LeadData(
            company_name=req.company_name,
            industry=req.industry,
            description=req.description,
            employee_count=req.employee_count,
            headquarters=req.headquarters,
            technologies=req.technologies or [],
            url=req.website,
            funding_info={
                "total_raised": req.funding_total,
                "last_round": req.funding_round,
            } if req.funding_total or req.funding_round else None,
        ))

    # Score batch
    result = default_engine.score_batch(
        leads=lead_objects,
        llm_for_top_n=use_llm_for_top,
        min_score_threshold=min_score,
    )

    # Format results
    formatted_results = []
    for r in result.results:
        formatted_results.append({
            "company_name": r.company_name,
            "score": r.scoring.overall_score,
            "grade": r.scoring.grade.value,
            "status": r.scoring.qualification_status.value,
            "priority": r.summary.get("priority_rank", "P3") if r.summary else "P4",
            "summary": r.summary.get("one_liner", "") if r.summary else "",
            "filter_passed": r.filter_status.passed,
            "has_llm_analysis": r.intelligence is not None,
        })

    return {
        "total_processed": result.processed,
        "qualified": result.qualified,
        "high_priority": result.high_priority,
        "rejected": result.rejected,
        "llm_enriched": result.llm_enriched,
        "processing_time_ms": result.processing_time_ms,
        "results": formatted_results,
    }


@app.post("/api/icp/explain", tags=["Scoring"])
async def explain_lead(request: FullScoreRequest):
    """
    Get detailed LLM explanation for a lead

    Always uses LLM for rich analysis including:
    - Match explanation
    - Talking points for sales
    - Risk identification
    - Recommended actions
    """
    # Force LLM
    request.use_llm = True
    return await score_lead(request)


# =============================================================================
# ICP Configuration Endpoints
# =============================================================================

@app.post("/api/icp/configure", tags=["Configuration"])
async def create_icp_config(config: ICPConfig):
    """Create or update an ICP configuration"""
    if not config.icp_id:
        config.icp_id = str(uuid.uuid4())

    icp_configs[config.icp_id] = config

    # Create engine with this config
    api_key = os.getenv("OPENROUTER_API_KEY")
    engines[config.icp_id] = ICPScoringEngine(
        icp_config=config,
        llm_api_key=api_key,
        llm_provider="openrouter"
    )

    return {
        "icp_id": config.icp_id,
        "status": "created",
        "message": "ICP configuration saved successfully",
    }


@app.get("/api/icp/configure", tags=["Configuration"])
async def list_configs():
    """List all ICP configurations"""
    return {
        "count": len(icp_configs),
        "configs": [
            {"icp_id": c.icp_id, "name": c.name, "created_at": c.created_at.isoformat()}
            for c in icp_configs.values()
        ],
    }


@app.get("/api/icp/configure/{icp_id}", tags=["Configuration"])
async def get_config(icp_id: str):
    """Get an ICP configuration by ID"""
    if icp_id not in icp_configs:
        raise HTTPException(status_code=404, detail="ICP config not found")
    return icp_configs[icp_id]


@app.delete("/api/icp/configure/{icp_id}", tags=["Configuration"])
async def delete_config(icp_id: str):
    """Delete an ICP configuration"""
    if icp_id not in icp_configs:
        raise HTTPException(status_code=404, detail="ICP config not found")
    del icp_configs[icp_id]
    if icp_id in engines:
        del engines[icp_id]
    return {"status": "deleted", "icp_id": icp_id}


# =============================================================================
# Statistics & Testing
# =============================================================================

@app.get("/api/stats", tags=["Info"])
async def get_stats():
    """Get engine statistics"""
    return {
        "default_engine": default_engine.get_stats(),
        "custom_configs": len(icp_configs),
    }


@app.post("/api/test", tags=["Testing"])
async def test_api(use_llm: bool = Query(False, description="Include LLM analysis")):
    """
    Test the API with sample data

    Use this to verify the API is working correctly.
    """
    sample_lead = LeadData(
        company_name="TestCorp AI",
        industry="SaaS",
        description="B2B SaaS company building AI-powered analytics platform. "
                    "Recently raised Series A funding. Actively hiring engineers.",
        employee_count=75,
        headquarters="San Francisco, CA",
        technologies=["AWS", "Python", "React", "PostgreSQL"],
        funding_info={"total_raised": 10000000, "last_round": "Series A"},
        job_postings=["Senior Engineer", "ML Engineer"],
    )

    result = default_engine.score_lead(sample_lead, skip_llm=not use_llm, force_llm=use_llm)

    response = {
        "test": "success",
        "company": result.company_name,
        "score": result.scoring.overall_score,
        "grade": result.scoring.grade.value,
        "status": result.scoring.qualification_status.value,
        "processing_time_ms": result.total_processing_time_ms,
    }

    if result.intelligence and use_llm:
        response["llm_analysis"] = {
            "relationship": result.intelligence.relationship_type.value,
            "explanation": result.intelligence.match_explanation,
            "talking_points": result.intelligence.talking_points[:3],
        }

    return response


# =============================================================================
# Helper Functions
# =============================================================================

def _get_engine(icp_id: Optional[str] = None) -> ICPScoringEngine:
    """Get engine by ICP ID or return default"""
    if icp_id and icp_id in engines:
        return engines[icp_id]
    return default_engine


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "type": type(exc).__name__,
        }
    )
