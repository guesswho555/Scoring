"""
ICP Scoring Engine - Main Orchestrator
======================================
Orchestrates the four-stage pipeline:
  Stage 1: Hard Filters → Stage 2: Signal Extraction →
  Stage 3: Weighted Scoring → Stage 4: LLM Intelligence

Key optimizations:
- Early rejection at Stage 1 saves processing costs
- LLM only runs for qualified leads (above threshold)
- Batch processing support with parallel execution
"""

import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models.schemas import (
    LeadData,
    UserCompanyProfile,
    ICPScoreResult,
    FilterResult,
    SignalResult,
    ScoringResult,
    QualificationStatus,
    BatchScoreResult,
)
from .models.icp_config import ICPConfig, create_default_icp_config
from .stages.stage1_filters import HardFilterStage
from .stages.stage2_signals import SignalExtractionStage
from .stages.stage3_scoring import WeightedScoringStage
from .stages.stage4_llm import LLMIntelligenceStage


class ICPScoringEngine:
    """
    Main ICP Scoring Engine that orchestrates all four stages.
    """

    def __init__(
        self,
        icp_config: Optional[ICPConfig] = None,
        llm_api_key: Optional[str] = None,
        llm_provider: str = "openai",
    ):
        """
        Initialize the scoring engine.

        Args:
            icp_config: ICP configuration (uses defaults if not provided)
            llm_api_key: API key for LLM provider
            llm_provider: LLM provider ("openai" or "anthropic")
        """
        self.config = icp_config or create_default_icp_config()

        # Initialize stages
        self.stage1 = HardFilterStage(self.config)
        self.stage2 = SignalExtractionStage()
        self.stage3 = WeightedScoringStage(self.config)
        self.stage4 = LLMIntelligenceStage(api_key=llm_api_key, provider=llm_provider)

        # Track statistics
        self.stats = {
            "total_processed": 0,
            "stage1_rejected": 0,
            "stage4_enriched": 0,
            "total_processing_time_ms": 0,
        }

    def score_lead(
        self,
        lead: LeadData,
        user_company: Optional[UserCompanyProfile] = None,
        skip_llm: bool = False,
        force_llm: bool = False,
    ) -> ICPScoreResult:
        """
        Score a single lead through the four-stage pipeline.

        Args:
            lead: Lead data to score
            user_company: User's company profile for comparison
            skip_llm: Skip Stage 4 even if lead qualifies
            force_llm: Force Stage 4 even if lead doesn't qualify

        Returns:
            Complete ICPScoreResult with all stage outputs
        """
        start_time = time.time()
        self.stats["total_processed"] += 1

        # =====================================================================
        # STAGE 1: Hard Filters
        # =====================================================================
        filter_result = self.stage1.process(lead)

        if not filter_result.passed:
            # Early rejection - return minimal result
            self.stats["stage1_rejected"] += 1
            return self._create_rejected_result(
                lead=lead,
                filter_result=filter_result,
                start_time=start_time,
            )

        # =====================================================================
        # STAGE 2: Signal Extraction
        # =====================================================================
        signal_result = self.stage2.process(lead)

        # =====================================================================
        # STAGE 3: Weighted Scoring
        # =====================================================================
        user_dict = user_company.model_dump() if user_company else None
        scoring_result = self.stage3.process(lead, signal_result, user_dict)

        # =====================================================================
        # STAGE 4: LLM Intelligence (conditional)
        # =====================================================================
        llm_result = None
        should_run_llm = (
            (scoring_result.proceed_to_llm and not skip_llm) or force_llm
        )

        if should_run_llm:
            self.stats["stage4_enriched"] += 1
            llm_result = self.stage4.process(
                lead=lead,
                signals=signal_result,
                scoring=scoring_result,
                user_company=user_company,
            )

        # =====================================================================
        # Assemble Final Result
        # =====================================================================
        total_time = (time.time() - start_time) * 1000
        self.stats["total_processing_time_ms"] += total_time

        summary = self._generate_summary(lead, scoring_result, llm_result)

        return ICPScoreResult(
            lead_id=lead.lead_id,
            company_name=lead.company_name,
            processed_at=datetime.utcnow(),
            filter_status=filter_result,
            signals=signal_result,
            scoring=scoring_result,
            intelligence=llm_result,
            summary=summary,
            total_processing_time_ms=round(total_time, 2),
        )

    def score_batch(
        self,
        leads: List[LeadData],
        user_company: Optional[UserCompanyProfile] = None,
        llm_for_top_n: int = 20,
        min_score_threshold: int = 50,
        max_workers: int = 4,
        sort_by: str = "score",
    ) -> BatchScoreResult:
        """
        Score multiple leads efficiently.

        Args:
            leads: List of leads to score
            user_company: User's company profile
            llm_for_top_n: Only run LLM for top N scoring leads
            min_score_threshold: Minimum score for inclusion
            max_workers: Number of parallel workers
            sort_by: Sort results by "score", "company_name", or "processing_time"

        Returns:
            BatchScoreResult with all results and statistics
        """
        start_time = time.time()
        results = []

        # First pass: Score all leads without LLM
        def score_without_llm(lead):
            return self.score_lead(lead, user_company, skip_llm=True)

        # Parallel scoring
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(score_without_llm, lead): lead
                for lead in leads
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    lead = futures[future]
                    # Create error result
                    results.append(self._create_error_result(lead, str(e)))

        # Sort by score (descending)
        results.sort(key=lambda x: x.scoring.overall_score, reverse=True)

        # Second pass: Run LLM for top N qualified leads
        llm_enriched = 0
        for i, result in enumerate(results):
            if i >= llm_for_top_n:
                break
            if (
                result.filter_status.passed
                and result.scoring.overall_score >= min_score_threshold
                and result.scoring.proceed_to_llm
            ):
                # Re-run with LLM
                lead = self._find_lead_by_name(leads, result.company_name)
                if lead:
                    enriched = self.score_lead(lead, user_company, force_llm=True)
                    results[i] = enriched
                    llm_enriched += 1

        # Apply final sorting
        if sort_by == "score":
            results.sort(key=lambda x: x.scoring.overall_score, reverse=True)
        elif sort_by == "company_name":
            results.sort(key=lambda x: x.company_name)
        elif sort_by == "processing_time":
            results.sort(key=lambda x: x.total_processing_time_ms)

        # Calculate statistics
        total_time = (time.time() - start_time) * 1000
        qualified = sum(
            1 for r in results
            if r.scoring.qualification_status in [
                QualificationStatus.HIGH_PRIORITY,
                QualificationStatus.AUTO_QUALIFIED,
            ]
        )
        high_priority = sum(
            1 for r in results
            if r.scoring.qualification_status == QualificationStatus.HIGH_PRIORITY
        )
        rejected = sum(
            1 for r in results
            if not r.filter_status.passed
            or r.scoring.qualification_status == QualificationStatus.REJECTED
        )

        return BatchScoreResult(
            processed=len(leads),
            qualified=qualified,
            high_priority=high_priority,
            rejected=rejected,
            llm_enriched=llm_enriched,
            processing_time_ms=round(total_time, 2),
            results=results,
        )

    def update_config(self, new_config: ICPConfig):
        """Update the ICP configuration and reinitialize stages"""
        self.config = new_config
        self.stage1 = HardFilterStage(new_config)
        self.stage3 = WeightedScoringStage(new_config)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = self.stats.copy()
        if stats["total_processed"] > 0:
            stats["stage1_rejection_rate"] = round(
                stats["stage1_rejected"] / stats["total_processed"] * 100, 1
            )
            stats["llm_enrichment_rate"] = round(
                stats["stage4_enriched"] / stats["total_processed"] * 100, 1
            )
            stats["avg_processing_time_ms"] = round(
                stats["total_processing_time_ms"] / stats["total_processed"], 2
            )
        return stats

    def reset_stats(self):
        """Reset engine statistics"""
        self.stats = {
            "total_processed": 0,
            "stage1_rejected": 0,
            "stage4_enriched": 0,
            "total_processing_time_ms": 0,
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _create_rejected_result(
        self,
        lead: LeadData,
        filter_result: FilterResult,
        start_time: float,
    ) -> ICPScoreResult:
        """Create a result for a rejected lead"""
        from .models.schemas import (
            SignalResult,
            ScoringResult,
            ScoringBreakdown,
            TierScore,
            Grade,
        )

        total_time = (time.time() - start_time) * 1000

        # Empty signal result
        empty_signals = SignalResult(
            positive=[],
            negative=[],
            total_boost=0,
            total_penalty=0,
            net_signal_score=0,
            processing_time_ms=0,
        )

        # Zero score result
        zero_tier = TierScore(score=0, weighted=0, details={})
        empty_scoring = ScoringResult(
            overall_score=0,
            grade=Grade.F,
            qualification_status=QualificationStatus.REJECTED,
            breakdown=ScoringBreakdown(
                firmographic=zero_tier,
                technographic=zero_tier,
                behavioral=zero_tier,
                signal_boost=zero_tier,
            ),
            data_completeness=0,
            proceed_to_llm=False,
            processing_time_ms=0,
        )

        return ICPScoreResult(
            lead_id=lead.lead_id,
            company_name=lead.company_name,
            processed_at=datetime.utcnow(),
            filter_status=filter_result,
            signals=empty_signals,
            scoring=empty_scoring,
            intelligence=None,
            summary={
                "one_liner": f"Rejected: {filter_result.rejection_reason}",
                "priority_rank": "REJECTED",
                "next_step": "No action required",
            },
            total_processing_time_ms=round(total_time, 2),
        )

    def _create_error_result(self, lead: LeadData, error: str) -> ICPScoreResult:
        """Create a result for a processing error"""
        from .models.schemas import (
            SignalResult,
            ScoringResult,
            ScoringBreakdown,
            TierScore,
            Grade,
        )

        error_filter = FilterResult(
            passed=False,
            checked_filters=0,
            rejection_reason=f"Processing error: {error[:100]}",
            rejected_by="system_error",
            processing_time_ms=0,
        )

        zero_tier = TierScore(score=0, weighted=0, details={})
        empty_signals = SignalResult()
        empty_scoring = ScoringResult(
            overall_score=0,
            grade=Grade.F,
            qualification_status=QualificationStatus.REJECTED,
            breakdown=ScoringBreakdown(
                firmographic=zero_tier,
                technographic=zero_tier,
                behavioral=zero_tier,
                signal_boost=zero_tier,
            ),
            data_completeness=0,
            proceed_to_llm=False,
        )

        return ICPScoreResult(
            lead_id=lead.lead_id,
            company_name=lead.company_name,
            filter_status=error_filter,
            signals=empty_signals,
            scoring=empty_scoring,
            summary={"one_liner": f"Error: {error[:50]}", "priority_rank": "ERROR"},
        )

    def _generate_summary(
        self,
        lead: LeadData,
        scoring: ScoringResult,
        llm_result: Optional[Any],
    ) -> Dict[str, str]:
        """Generate a human-readable summary"""
        score = scoring.overall_score
        grade = scoring.grade.value
        status = scoring.qualification_status.value

        # Priority rank
        if status == "HIGH_PRIORITY":
            priority = "P1"
        elif status == "AUTO_QUALIFIED":
            priority = "P2"
        elif status == "REVIEW":
            priority = "P3"
        else:
            priority = "P4"

        # One-liner
        if llm_result and llm_result.match_explanation:
            one_liner = llm_result.match_explanation[:100]
        else:
            one_liner = f"{lead.company_name}: Score {score}/100 (Grade {grade})"

        # Next step
        if llm_result:
            next_step = llm_result.recommended_action.replace("_", " ").title()
        elif score >= 80:
            next_step = "High priority outreach"
        elif score >= 60:
            next_step = "Standard outreach"
        elif score >= 40:
            next_step = "Add to nurture sequence"
        else:
            next_step = "Review or disqualify"

        return {
            "one_liner": one_liner,
            "priority_rank": priority,
            "next_step": next_step,
        }

    def _find_lead_by_name(
        self, leads: List[LeadData], company_name: str
    ) -> Optional[LeadData]:
        """Find a lead by company name"""
        for lead in leads:
            if lead.company_name == company_name:
                return lead
        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def create_engine(
    target_industries: Optional[List[str]] = None,
    target_tech: Optional[List[str]] = None,
    target_regions: Optional[List[str]] = None,
    llm_api_key: Optional[str] = None,
) -> ICPScoringEngine:
    """
    Factory function to create an ICP Scoring Engine with common settings.

    Args:
        target_industries: List of target industries
        target_tech: List of target technologies
        target_regions: List of target regions
        llm_api_key: API key for LLM provider

    Returns:
        Configured ICPScoringEngine instance
    """
    config = create_default_icp_config(
        target_industries=target_industries,
        target_tech=target_tech,
        target_regions=target_regions,
    )
    return ICPScoringEngine(icp_config=config, llm_api_key=llm_api_key)


def quick_score(lead_data: Dict[str, Any]) -> ICPScoreResult:
    """
    Quick scoring function for a single lead.

    Args:
        lead_data: Dictionary with lead information

    Returns:
        ICPScoreResult
    """
    engine = ICPScoringEngine()
    lead = LeadData(**lead_data)
    return engine.score_lead(lead, skip_llm=True)
