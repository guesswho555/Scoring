"""
Pydantic schemas for ICP Scoring Engine
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# ENUMS
# =============================================================================

class RelationshipType(str, Enum):
    """Classification of relationship between user and lead company"""
    BUYER = "Buyer"
    PARTNER = "Partner"
    COMPETITOR = "Competitor"
    INVESTOR = "Investor"
    NO_FIT = "No_Fit"


class QualificationStatus(str, Enum):
    """Lead qualification status"""
    HIGH_PRIORITY = "HIGH_PRIORITY"
    AUTO_QUALIFIED = "AUTO_QUALIFIED"
    REVIEW = "REVIEW"
    REJECTED = "REJECTED"


class Grade(str, Enum):
    """ICP score grade"""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class Severity(str, Enum):
    """Risk severity level"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# =============================================================================
# INPUT SCHEMAS
# =============================================================================

class FundingInfo(BaseModel):
    """Funding information for a company"""
    total_raised: Optional[float] = None
    last_round: Optional[str] = None
    last_round_amount: Optional[float] = None
    investors: List[str] = Field(default_factory=list)


class SocialLinks(BaseModel):
    """Social media links"""
    linkedin: Optional[str] = None
    twitter: Optional[str] = None
    facebook: Optional[str] = None
    crunchbase: Optional[str] = None


class LeadData(BaseModel):
    """Input schema for lead data (from Phase 4 scrape)"""
    lead_id: Optional[str] = None
    company_name: str
    url: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    raw_content: Optional[str] = None
    meta_tags: List[str] = Field(default_factory=list)
    employee_count: Optional[int] = None
    employee_range: Optional[str] = None
    founded_year: Optional[int] = None
    headquarters: Optional[str] = None
    country: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    funding_info: Optional[FundingInfo] = None
    social_links: Optional[SocialLinks] = None
    recent_news: List[str] = Field(default_factory=list)
    job_postings: List[str] = Field(default_factory=list)
    scrape_timestamp: Optional[datetime] = None

    class Config:
        extra = "allow"


class UserCompanyProfile(BaseModel):
    """User's company profile for comparison"""
    company_id: Optional[str] = None
    name: str
    core_offering: str
    value_proposition: Optional[str] = None
    target_industries: List[str] = Field(default_factory=list)
    target_company_sizes: Optional[Dict[str, int]] = None
    technologies: List[str] = Field(default_factory=list)
    competitors: List[str] = Field(default_factory=list)


# =============================================================================
# STAGE RESULT SCHEMAS
# =============================================================================

class FilterResult(BaseModel):
    """Result from Stage 1: Hard Filters"""
    passed: bool
    checked_filters: int = 0
    rejection_reason: Optional[str] = None
    rejected_by: Optional[str] = None
    processing_time_ms: float = 0


class ExtractedSignal(BaseModel):
    """A single extracted signal"""
    tag: str
    category: str
    weight: int
    matched_text: Optional[str] = None
    pattern: Optional[str] = None


class SignalResult(BaseModel):
    """Result from Stage 2: Signal Extraction"""
    positive: List[ExtractedSignal] = Field(default_factory=list)
    negative: List[ExtractedSignal] = Field(default_factory=list)
    total_boost: int = 0
    total_penalty: int = 0
    net_signal_score: int = 0
    processing_time_ms: float = 0


class TierScore(BaseModel):
    """Score for a single tier"""
    score: float
    weighted: float
    details: Dict[str, Any] = Field(default_factory=dict)


class ScoringBreakdown(BaseModel):
    """Breakdown of scores by tier"""
    firmographic: TierScore
    technographic: TierScore
    behavioral: TierScore
    signal_boost: TierScore


class ScoringResult(BaseModel):
    """Result from Stage 3: Weighted Scoring"""
    overall_score: int
    grade: Grade
    qualification_status: QualificationStatus
    breakdown: ScoringBreakdown
    data_completeness: int
    proceed_to_llm: bool
    processing_time_ms: float = 0


class Risk(BaseModel):
    """Identified risk factor"""
    risk: str
    severity: Severity
    mitigation: Optional[str] = None


class LLMResult(BaseModel):
    """Result from Stage 4: LLM Intelligence"""
    relationship_type: RelationshipType
    relationship_confidence: int = 0
    match_explanation: str
    talking_points: List[str] = Field(default_factory=list)
    risks: List[Risk] = Field(default_factory=list)
    recommended_action: str
    suggested_channel: Optional[str] = None
    urgency: Optional[str] = None
    llm_confidence: int = 0
    processing_time_ms: float = 0
    tokens_used: Optional[int] = None


# =============================================================================
# UNIFIED OUTPUT SCHEMA
# =============================================================================

class ICPScoreResult(BaseModel):
    """Complete ICP scoring result combining all stages"""
    lead_id: Optional[str] = None
    company_name: str
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    # Stage 1: Filter Result
    filter_status: FilterResult

    # Stage 2: Signal Result
    signals: SignalResult

    # Stage 3: Scoring Result
    scoring: ScoringResult

    # Stage 4: LLM Result (optional - only for qualified leads)
    intelligence: Optional[LLMResult] = None

    # Summary
    summary: Optional[Dict[str, str]] = None

    # Total processing time
    total_processing_time_ms: float = 0


# =============================================================================
# API REQUEST SCHEMAS
# =============================================================================

class ScoreRequest(BaseModel):
    """Request to score a single lead"""
    icp_id: Optional[str] = None
    user_company_id: Optional[str] = None
    lead_data: LeadData
    user_company: Optional[UserCompanyProfile] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchScoreRequest(BaseModel):
    """Request to score multiple leads"""
    icp_id: Optional[str] = None
    user_company_id: Optional[str] = None
    leads: List[LeadData]
    user_company: Optional[UserCompanyProfile] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchScoreResult(BaseModel):
    """Result from batch scoring"""
    processed: int
    qualified: int
    high_priority: int
    rejected: int
    llm_enriched: int
    processing_time_ms: float
    results: List[ICPScoreResult]


class PipelineRequest(BaseModel):
    """Request for full pipeline (scrape -> score)"""
    url: str
    user_company_id: str
    user_company: Optional[UserCompanyProfile] = None
    icp_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
