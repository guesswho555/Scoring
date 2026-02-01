"""
Stage 3: Weighted Scoring
=========================
Deterministic multi-tier scoring based on configurable weights.

Tiers:
- Firmographic (35%): Industry, size, revenue, geography
- Technographic (25%): Tech stack overlap, platform compatibility
- Behavioral (20%): Growth signals, digital presence
- Signal Boost (20%): From Stage 2 signal extraction
"""

import time
from typing import Optional, Dict, Any, List

from ..models.schemas import (
    LeadData,
    SignalResult,
    ScoringResult,
    ScoringBreakdown,
    TierScore,
    Grade,
    QualificationStatus,
)
from ..models.icp_config import ICPConfig
from ..config.settings import (
    DEFAULT_WEIGHTS,
    DEFAULT_THRESHOLDS,
    GRADE_MAPPING,
    INDUSTRY_CATEGORIES,
    TECH_CATEGORIES,
)


class WeightedScoringStage:
    """
    Stage 3: Calculate weighted ICP score across multiple tiers.
    """

    def __init__(self, config: Optional[ICPConfig] = None):
        """
        Initialize with ICP configuration or use defaults.
        """
        self.config = config
        if config:
            self.weights = {
                "firmographic": config.weights.firmographic,
                "technographic": config.weights.technographic,
                "behavioral": config.weights.behavioral,
                "signal_boost": config.weights.signal_boost,
            }
            self.thresholds = {
                "auto_qualify": config.thresholds.auto_qualify,
                "review_min": config.thresholds.review_min,
                "review_max": config.thresholds.review_max,
                "auto_reject": config.thresholds.auto_reject,
                "llm_enrichment_threshold": config.thresholds.llm_enrichment_threshold,
            }
        else:
            self.weights = DEFAULT_WEIGHTS
            self.thresholds = DEFAULT_THRESHOLDS

    def process(
        self,
        lead: LeadData,
        signals: SignalResult,
        user_company: Optional[Dict] = None,
    ) -> ScoringResult:
        """
        Calculate weighted ICP score.

        Args:
            lead: Lead data to score
            signals: Signal extraction results from Stage 2
            user_company: Optional user company profile for comparison

        Returns:
            ScoringResult with overall score, grade, and breakdown
        """
        start_time = time.time()

        # Calculate tier scores
        firmographic = self._calculate_firmographic_score(lead)
        technographic = self._calculate_technographic_score(lead, user_company)
        behavioral = self._calculate_behavioral_score(lead, signals)
        signal_boost = self._calculate_signal_boost_score(signals)

        # Calculate weighted contributions
        firm_weighted = firmographic["score"] * self.weights["firmographic"]
        tech_weighted = technographic["score"] * self.weights["technographic"]
        behav_weighted = behavioral["score"] * self.weights["behavioral"]
        signal_weighted = signal_boost["score"] * self.weights["signal_boost"]

        # Overall score
        overall = firm_weighted + tech_weighted + behav_weighted + signal_weighted
        overall = max(0, min(100, round(overall)))

        # Determine grade and qualification
        grade = self._map_to_grade(overall)
        qualification = self._determine_qualification(overall)
        proceed_to_llm = overall >= self.thresholds["llm_enrichment_threshold"]

        # Data completeness
        completeness = self._calculate_data_completeness(lead)

        processing_time = (time.time() - start_time) * 1000

        return ScoringResult(
            overall_score=overall,
            grade=grade,
            qualification_status=qualification,
            breakdown=ScoringBreakdown(
                firmographic=TierScore(
                    score=firmographic["score"],
                    weighted=round(firm_weighted, 2),
                    details=firmographic["details"],
                ),
                technographic=TierScore(
                    score=technographic["score"],
                    weighted=round(tech_weighted, 2),
                    details=technographic["details"],
                ),
                behavioral=TierScore(
                    score=behavioral["score"],
                    weighted=round(behav_weighted, 2),
                    details=behavioral["details"],
                ),
                signal_boost=TierScore(
                    score=signal_boost["score"],
                    weighted=round(signal_weighted, 2),
                    details=signal_boost["details"],
                ),
            ),
            data_completeness=completeness,
            proceed_to_llm=proceed_to_llm,
            processing_time_ms=round(processing_time, 2),
        )

    def _calculate_firmographic_score(self, lead: LeadData) -> Dict[str, Any]:
        """Calculate firmographic tier score (industry, size, revenue, geography)"""
        details = {}
        scores = []

        # Industry match (40% of tier)
        industry_score = self._score_industry(lead)
        details["industry_match"] = industry_score
        scores.append(industry_score["score"] * 0.40)

        # Company size (25% of tier)
        size_score = self._score_company_size(lead)
        details["company_size"] = size_score
        scores.append(size_score["score"] * 0.25)

        # Revenue/Funding (20% of tier)
        revenue_score = self._score_revenue(lead)
        details["revenue_fit"] = revenue_score
        scores.append(revenue_score["score"] * 0.20)

        # Geography (15% of tier)
        geo_score = self._score_geography(lead)
        details["geography"] = geo_score
        scores.append(geo_score["score"] * 0.15)

        total_score = sum(scores)
        return {"score": round(total_score, 1), "details": details}

    def _calculate_technographic_score(
        self, lead: LeadData, user_company: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Calculate technographic tier score (tech stack, platform compatibility)"""
        details = {}
        scores = []

        # Tech stack overlap (60% of tier)
        tech_score = self._score_tech_stack(lead, user_company)
        details["tech_stack"] = tech_score
        scores.append(tech_score["score"] * 0.60)

        # Platform compatibility (40% of tier)
        platform_score = self._score_platform_compatibility(lead)
        details["platform"] = platform_score
        scores.append(platform_score["score"] * 0.40)

        total_score = sum(scores)
        return {"score": round(total_score, 1), "details": details}

    def _calculate_behavioral_score(
        self, lead: LeadData, signals: SignalResult
    ) -> Dict[str, Any]:
        """Calculate behavioral tier score (growth, presence)"""
        details = {}
        scores = []

        # Growth signals (60% of tier)
        growth_score = self._score_growth_signals(lead, signals)
        details["growth"] = growth_score
        scores.append(growth_score["score"] * 0.60)

        # Digital presence (40% of tier)
        presence_score = self._score_digital_presence(lead)
        details["digital_presence"] = presence_score
        scores.append(presence_score["score"] * 0.40)

        total_score = sum(scores)
        return {"score": round(total_score, 1), "details": details}

    def _calculate_signal_boost_score(self, signals: SignalResult) -> Dict[str, Any]:
        """Calculate signal boost tier score from Stage 2"""
        # Base score of 50 (neutral)
        base_score = 50

        # Apply boost and penalty (capped)
        boost = min(signals.total_boost, 40)  # Cap at +40
        penalty = min(signals.total_penalty, 40)  # Cap at -40

        score = base_score + boost - penalty
        score = max(0, min(100, score))

        details = {
            "base": base_score,
            "boost_applied": boost,
            "penalty_applied": penalty,
            "positive_signals": len(signals.positive),
            "negative_signals": len(signals.negative),
        }

        return {"score": round(score, 1), "details": details}

    # =========================================================================
    # Sub-scoring functions
    # =========================================================================

    def _score_industry(self, lead: LeadData) -> Dict[str, Any]:
        """Score industry match"""
        if not lead.industry:
            return {"score": 50, "reason": "No industry data"}

        lead_industry = lead.industry.lower()

        # Check if we have ICP config with target industries
        if self.config and self.config.firmographic_criteria.target_industries.primary:
            targets = self.config.firmographic_criteria.target_industries
            primary = [i.lower() for i in targets.primary]
            secondary = [i.lower() for i in targets.secondary]

            # Check primary match
            for industry in primary:
                if industry in lead_industry or lead_industry in industry:
                    return {
                        "score": targets.weight_primary,
                        "reason": f"Primary industry match: {lead.industry}",
                    }

            # Check secondary match
            for industry in secondary:
                if industry in lead_industry or lead_industry in industry:
                    return {
                        "score": targets.weight_secondary,
                        "reason": f"Secondary industry match: {lead.industry}",
                    }

            return {"score": 30, "reason": f"No industry match: {lead.industry}"}

        # Default scoring based on common tech industries
        tech_industries = ["saas", "software", "technology", "tech", "fintech", "b2b"]
        for tech in tech_industries:
            if tech in lead_industry:
                return {"score": 80, "reason": f"Tech industry: {lead.industry}"}

        return {"score": 50, "reason": f"Neutral industry: {lead.industry}"}

    def _score_company_size(self, lead: LeadData) -> Dict[str, Any]:
        """Score company size fit"""
        if not lead.employee_count:
            # Try to parse from employee_range
            if lead.employee_range:
                lead.employee_count = self._parse_employee_range(lead.employee_range)

        if not lead.employee_count:
            return {"score": 50, "reason": "No size data"}

        count = lead.employee_count

        if self.config:
            size = self.config.firmographic_criteria.company_size
            ideal_min, ideal_max = size.ideal_min, size.ideal_max
            accept_min, accept_max = size.acceptable_min, size.acceptable_max
        else:
            ideal_min, ideal_max = 50, 500
            accept_min, accept_max = 20, 1000

        # Score based on fit
        if ideal_min <= count <= ideal_max:
            return {"score": 100, "reason": f"Ideal size: {count} employees"}
        elif accept_min <= count <= accept_max:
            return {"score": 70, "reason": f"Acceptable size: {count} employees"}
        elif count < accept_min:
            return {"score": 40, "reason": f"Too small: {count} employees"}
        else:
            return {"score": 50, "reason": f"Large enterprise: {count} employees"}

    def _score_revenue(self, lead: LeadData) -> Dict[str, Any]:
        """Score revenue/funding fit"""
        if not lead.funding_info:
            return {"score": 50, "reason": "No funding data"}

        funding = lead.funding_info
        if not funding.total_raised:
            return {"score": 50, "reason": "No funding amount"}

        amount = funding.total_raised

        # Score based on funding (as proxy for revenue/budget)
        if amount >= 50_000_000:
            return {"score": 100, "reason": f"Well-funded: ${amount:,.0f}"}
        elif amount >= 10_000_000:
            return {"score": 90, "reason": f"Strong funding: ${amount:,.0f}"}
        elif amount >= 5_000_000:
            return {"score": 80, "reason": f"Good funding: ${amount:,.0f}"}
        elif amount >= 1_000_000:
            return {"score": 70, "reason": f"Seed/Early: ${amount:,.0f}"}
        else:
            return {"score": 50, "reason": f"Bootstrapped: ${amount:,.0f}"}

    def _score_geography(self, lead: LeadData) -> Dict[str, Any]:
        """Score geographic fit"""
        location = lead.headquarters or lead.country
        if not location:
            return {"score": 50, "reason": "No location data"}

        location_lower = location.lower()

        if self.config:
            geo = self.config.firmographic_criteria.geography
            preferred = [g.lower() for g in geo.preferred]
            acceptable = [g.lower() for g in geo.acceptable]

            for region in preferred:
                if region in location_lower:
                    return {
                        "score": geo.weight_preferred,
                        "reason": f"Preferred region: {location}",
                    }

            for region in acceptable:
                if region in location_lower:
                    return {
                        "score": geo.weight_acceptable,
                        "reason": f"Acceptable region: {location}",
                    }

        # Default: US/UK/EU preferred
        preferred_regions = ["united states", "usa", "us", "uk", "united kingdom", "europe", "eu", "canada", "australia"]
        for region in preferred_regions:
            if region in location_lower:
                return {"score": 80, "reason": f"Target region: {location}"}

        return {"score": 50, "reason": f"Other region: {location}"}

    def _score_tech_stack(
        self, lead: LeadData, user_company: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Score technology stack overlap"""
        if not lead.technologies:
            return {"score": 50, "reason": "No tech data", "matched": []}

        lead_tech = [t.lower() for t in lead.technologies]

        # Get target tech from config or user company
        target_tech = []
        if self.config:
            target_tech = [t.lower() for t in self.config.technographic_criteria.preferred_tech]
            required = [t.lower() for t in self.config.technographic_criteria.required_tech]
            target_tech.extend(required)
        if user_company and user_company.get("technologies"):
            target_tech.extend([t.lower() for t in user_company["technologies"]])

        if not target_tech:
            # Score based on modern tech stack
            modern_tech = []
            for category, techs in TECH_CATEGORIES.items():
                modern_tech.extend(techs)

            matched = [t for t in lead_tech if any(m in t for m in modern_tech)]
            score = min(100, 40 + len(matched) * 10)
            return {
                "score": score,
                "reason": f"Modern stack: {len(matched)} technologies",
                "matched": matched[:5],
            }

        # Calculate overlap
        matched = []
        for lt in lead_tech:
            for tt in target_tech:
                if tt in lt or lt in tt:
                    matched.append(lt)
                    break

        if not target_tech:
            overlap_pct = 50
        else:
            overlap_pct = (len(matched) / len(target_tech)) * 100

        score = min(100, max(20, overlap_pct))
        return {
            "score": round(score, 1),
            "reason": f"{len(matched)}/{len(target_tech)} tech overlap",
            "matched": matched,
        }

    def _score_platform_compatibility(self, lead: LeadData) -> Dict[str, Any]:
        """Score platform/integration compatibility"""
        if not lead.technologies:
            return {"score": 50, "reason": "No tech data"}

        lead_tech = [t.lower() for t in lead.technologies]

        # Check for cloud/modern platform indicators
        cloud_indicators = TECH_CATEGORIES.get("cloud", [])
        api_indicators = ["api", "rest", "graphql", "webhook", "integration"]

        has_cloud = any(
            any(c in t for c in cloud_indicators) for t in lead_tech
        )
        has_api = any(
            any(a in t for a in api_indicators) for t in lead_tech
        )

        if has_cloud and has_api:
            return {"score": 100, "reason": "Cloud-native with API capability"}
        elif has_cloud:
            return {"score": 80, "reason": "Cloud-based platform"}
        elif has_api:
            return {"score": 70, "reason": "API-ready"}
        else:
            return {"score": 50, "reason": "Traditional platform"}

    def _score_growth_signals(
        self, lead: LeadData, signals: SignalResult
    ) -> Dict[str, Any]:
        """Score growth signals"""
        growth_tags = ["FUNDED", "GROWING", "SCALING", "HIRING", "IPO", "M&A", "GROWTH_STAGE"]

        growth_signals = [
            s for s in signals.positive if s.tag in growth_tags
        ]

        if not growth_signals:
            # Check job postings as backup
            if lead.job_postings and len(lead.job_postings) > 3:
                return {"score": 70, "reason": f"Actively hiring: {len(lead.job_postings)} positions"}
            return {"score": 50, "reason": "No growth signals detected"}

        # Score based on strongest growth signals
        score = 50 + min(50, len(growth_signals) * 15)
        reasons = [s.tag for s in growth_signals[:3]]

        return {
            "score": min(100, score),
            "reason": f"Growth signals: {', '.join(reasons)}",
            "signals": [s.tag for s in growth_signals],
        }

    def _score_digital_presence(self, lead: LeadData) -> Dict[str, Any]:
        """Score digital presence strength"""
        score = 50
        reasons = []

        # Check social links
        if lead.social_links:
            if lead.social_links.linkedin:
                score += 15
                reasons.append("LinkedIn")
            if lead.social_links.twitter:
                score += 10
                reasons.append("Twitter")
            if lead.social_links.crunchbase:
                score += 10
                reasons.append("Crunchbase")

        # Check for website
        if lead.url:
            score += 10
            reasons.append("Website")

        # Check for news/content
        if lead.recent_news and len(lead.recent_news) > 0:
            score += 10
            reasons.append("News coverage")

        score = min(100, score)
        reason = f"Presence: {', '.join(reasons)}" if reasons else "Limited presence"

        return {"score": score, "reason": reason}

    # =========================================================================
    # Helper functions
    # =========================================================================

    def _map_to_grade(self, score: int) -> Grade:
        """Map score to grade"""
        for min_score, max_score, grade, _ in GRADE_MAPPING:
            if min_score <= score <= max_score:
                return Grade(grade)
        return Grade.F

    def _determine_qualification(self, score: int) -> QualificationStatus:
        """Determine qualification status"""
        if score >= self.thresholds["auto_qualify"]:
            return QualificationStatus.HIGH_PRIORITY
        elif score >= self.thresholds["review_min"]:
            return QualificationStatus.AUTO_QUALIFIED if score >= 70 else QualificationStatus.REVIEW
        else:
            return QualificationStatus.REJECTED

    def _calculate_data_completeness(self, lead: LeadData) -> int:
        """Calculate data completeness percentage"""
        fields = [
            lead.company_name,
            lead.industry,
            lead.description,
            lead.employee_count,
            lead.headquarters,
            lead.technologies,
            lead.funding_info,
            lead.url,
            lead.social_links,
            lead.job_postings,
        ]

        populated = sum(1 for f in fields if f)
        return round((populated / len(fields)) * 100)

    def _parse_employee_range(self, range_str: str) -> Optional[int]:
        """Parse employee range string to approximate count"""
        import re

        # Try to extract numbers
        numbers = re.findall(r"\d+", range_str.replace(",", ""))
        if len(numbers) >= 2:
            # Return midpoint
            return (int(numbers[0]) + int(numbers[1])) // 2
        elif len(numbers) == 1:
            return int(numbers[0])
        return None
