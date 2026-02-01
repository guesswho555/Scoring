"""
ICP Configuration Models
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class StopList(BaseModel):
    """Stop list configuration for hard filters"""
    domains: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    industries: List[str] = Field(default_factory=list)


class DataRequirements(BaseModel):
    """Data completeness requirements"""
    min_fields_populated: int = 3
    required_fields: List[str] = Field(default_factory=lambda: ["company_name"])


class CompanySizeGate(BaseModel):
    """Company size filtering"""
    enabled: bool = False
    min_employees: int = 5
    max_employees: Optional[int] = None


class HardFiltersConfig(BaseModel):
    """Configuration for Stage 1: Hard Filters"""
    stop_list: StopList = Field(default_factory=StopList)
    data_requirements: DataRequirements = Field(default_factory=DataRequirements)
    geographic_exclusions: List[str] = Field(default_factory=list)
    company_size_gate: CompanySizeGate = Field(default_factory=CompanySizeGate)


class TargetIndustries(BaseModel):
    """Target industry configuration"""
    primary: List[str] = Field(default_factory=list)
    secondary: List[str] = Field(default_factory=list)
    weight_primary: int = 100
    weight_secondary: int = 70


class CompanySize(BaseModel):
    """Company size criteria"""
    ideal_min: int = 50
    ideal_max: int = 500
    acceptable_min: int = 20
    acceptable_max: int = 1000


class RevenueIndicators(BaseModel):
    """Revenue/funding indicators"""
    target_range: Optional[str] = None  # e.g., "1M-50M"
    funding_as_proxy: bool = True
    min_funding: Optional[float] = None
    max_funding: Optional[float] = None


class Geography(BaseModel):
    """Geographic targeting"""
    preferred: List[str] = Field(default_factory=list)
    acceptable: List[str] = Field(default_factory=list)
    weight_preferred: int = 100
    weight_acceptable: int = 60


class FirmographicCriteria(BaseModel):
    """Firmographic criteria for scoring"""
    target_industries: TargetIndustries = Field(default_factory=TargetIndustries)
    company_size: CompanySize = Field(default_factory=CompanySize)
    revenue_indicators: RevenueIndicators = Field(default_factory=RevenueIndicators)
    geography: Geography = Field(default_factory=Geography)


class TechnographicCriteria(BaseModel):
    """Technographic criteria for scoring"""
    required_tech: List[str] = Field(default_factory=list)
    preferred_tech: List[str] = Field(default_factory=list)
    competing_tech: List[str] = Field(default_factory=list)
    tech_match_scoring: str = "percentage_overlap"


class BehavioralCriteria(BaseModel):
    """Behavioral criteria for scoring"""
    growth_signals_weight: float = 1.5
    hiring_bonus: int = 10
    funding_bonus: int = 15
    digital_presence_minimum: int = 40


class ScoringWeights(BaseModel):
    """Weights for each scoring tier"""
    firmographic: float = 0.35
    technographic: float = 0.25
    behavioral: float = 0.20
    signal_boost: float = 0.20


class ScoringThresholds(BaseModel):
    """Thresholds for qualification"""
    auto_qualify: int = 80
    review_min: int = 50
    review_max: int = 79
    auto_reject: int = 49
    llm_enrichment_threshold: int = 60


class ICPConfig(BaseModel):
    """Complete ICP Configuration"""
    icp_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_company_id: Optional[str] = None
    name: str = "Default ICP"
    description: Optional[str] = None

    # Stage 1: Hard Filters
    hard_filters: HardFiltersConfig = Field(default_factory=HardFiltersConfig)

    # Stage 3: Scoring Criteria
    firmographic_criteria: FirmographicCriteria = Field(default_factory=FirmographicCriteria)
    technographic_criteria: TechnographicCriteria = Field(default_factory=TechnographicCriteria)
    behavioral_criteria: BehavioralCriteria = Field(default_factory=BehavioralCriteria)

    # Weights & Thresholds
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    thresholds: ScoringThresholds = Field(default_factory=ScoringThresholds)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    def update(self, **kwargs):
        """Update configuration and set updated_at"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
        return self


def create_default_icp_config(
    user_company_id: Optional[str] = None,
    target_industries: Optional[List[str]] = None,
    target_tech: Optional[List[str]] = None,
    target_regions: Optional[List[str]] = None,
) -> ICPConfig:
    """
    Factory function to create an ICP config with sensible defaults
    """
    config = ICPConfig(user_company_id=user_company_id)

    if target_industries:
        config.firmographic_criteria.target_industries.primary = target_industries

    if target_tech:
        config.technographic_criteria.preferred_tech = target_tech

    if target_regions:
        config.firmographic_criteria.geography.preferred = target_regions

    return config
