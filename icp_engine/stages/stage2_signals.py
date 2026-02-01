"""
Stage 2: Signal Extraction
==========================
Pattern matching to identify positive and negative signals in lead data.
These signals feed into Stage 3 scoring and Stage 4 LLM prompts.

Signals extracted:
- Growth indicators (funding, hiring, scaling)
- Need indicators (pain points, modernization)
- Fit indicators (B2B, tech stack)
- Red flags (layoffs, legal issues)
"""

import re
import time
from typing import List, Dict, Optional, Any

from ..models.schemas import LeadData, SignalResult, ExtractedSignal
from ..config.settings import SIGNAL_LIBRARY


class SignalExtractionStage:
    """
    Stage 2: Extract positive and negative signals from lead data.
    """

    def __init__(self, signal_library: Optional[Dict] = None):
        """
        Initialize with signal library or use defaults.
        """
        self.signals = signal_library or SIGNAL_LIBRARY
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance"""
        self.compiled_positive = {}
        self.compiled_negative = {}

        # Compile positive signal patterns
        for category, patterns in self.signals.get("positive_signals", {}).items():
            self.compiled_positive[category] = []
            for p in patterns:
                try:
                    compiled = re.compile(p["pattern"], re.IGNORECASE)
                    self.compiled_positive[category].append({
                        "regex": compiled,
                        "weight": p["weight"],
                        "tag": p["tag"],
                        "pattern": p["pattern"],
                    })
                except re.error:
                    # Skip invalid patterns
                    pass

        # Compile negative signal patterns
        for category, patterns in self.signals.get("negative_signals", {}).items():
            self.compiled_negative[category] = []
            for p in patterns:
                try:
                    compiled = re.compile(p["pattern"], re.IGNORECASE)
                    self.compiled_negative[category].append({
                        "regex": compiled,
                        "weight": p["weight"],
                        "tag": p["tag"],
                        "pattern": p["pattern"],
                    })
                except re.error:
                    pass

    def process(self, lead: LeadData) -> SignalResult:
        """
        Extract signals from lead data.

        Args:
            lead: Lead data to analyze

        Returns:
            SignalResult with positive/negative signals and scores
        """
        start_time = time.time()

        # Build text corpus for analysis
        text_corpus = self._build_text_corpus(lead)

        # Extract signals
        positive_signals = self._extract_positive_signals(text_corpus)
        negative_signals = self._extract_negative_signals(text_corpus)

        # Also check structured data
        structured_signals = self._extract_structured_signals(lead)
        positive_signals.extend(structured_signals["positive"])
        negative_signals.extend(structured_signals["negative"])

        # Deduplicate by tag (keep highest weight)
        positive_signals = self._deduplicate_signals(positive_signals)
        negative_signals = self._deduplicate_signals(negative_signals)

        # Calculate totals
        total_boost = sum(s.weight for s in positive_signals)
        total_penalty = sum(abs(s.weight) for s in negative_signals)
        net_score = total_boost - total_penalty

        processing_time = (time.time() - start_time) * 1000

        return SignalResult(
            positive=positive_signals,
            negative=negative_signals,
            total_boost=total_boost,
            total_penalty=total_penalty,
            net_signal_score=net_score,
            processing_time_ms=round(processing_time, 2),
        )

    def _build_text_corpus(self, lead: LeadData) -> str:
        """Combine all text fields for analysis"""
        parts = []

        if lead.description:
            parts.append(lead.description)
        if lead.raw_content:
            parts.append(lead.raw_content)
        if lead.meta_tags:
            parts.append(" ".join(lead.meta_tags))
        if lead.recent_news:
            parts.append(" ".join(lead.recent_news))
        if lead.job_postings:
            parts.append(" ".join(lead.job_postings))

        return " ".join(parts)

    def _extract_positive_signals(self, text: str) -> List[ExtractedSignal]:
        """Extract positive signals from text"""
        signals = []

        for category, patterns in self.compiled_positive.items():
            for p in patterns:
                match = p["regex"].search(text)
                if match:
                    # Get context around match
                    context = self._get_match_context(text, match, window=50)
                    signals.append(
                        ExtractedSignal(
                            tag=p["tag"],
                            category=category,
                            weight=p["weight"],
                            matched_text=context,
                            pattern=p["pattern"],
                        )
                    )

        return signals

    def _extract_negative_signals(self, text: str) -> List[ExtractedSignal]:
        """Extract negative signals from text"""
        signals = []

        for category, patterns in self.compiled_negative.items():
            for p in patterns:
                match = p["regex"].search(text)
                if match:
                    context = self._get_match_context(text, match, window=50)
                    signals.append(
                        ExtractedSignal(
                            tag=p["tag"],
                            category=category,
                            weight=p["weight"],  # Already negative in config
                            matched_text=context,
                            pattern=p["pattern"],
                        )
                    )

        return signals

    def _extract_structured_signals(self, lead: LeadData) -> Dict[str, List[ExtractedSignal]]:
        """Extract signals from structured data fields"""
        positive = []
        negative = []

        # Funding signals
        if lead.funding_info:
            funding = lead.funding_info
            if funding.total_raised and funding.total_raised > 0:
                weight = self._calculate_funding_weight(funding.total_raised)
                positive.append(
                    ExtractedSignal(
                        tag="FUNDED",
                        category="funding",
                        weight=weight,
                        matched_text=f"Raised ${funding.total_raised:,.0f}",
                    )
                )
            if funding.last_round:
                round_lower = funding.last_round.lower()
                if any(r in round_lower for r in ["series a", "series b", "series c", "series d"]):
                    positive.append(
                        ExtractedSignal(
                            tag="GROWTH_STAGE",
                            category="funding",
                            weight=12,
                            matched_text=f"Funding round: {funding.last_round}",
                        )
                    )

        # Hiring signals (from job postings)
        if lead.job_postings and len(lead.job_postings) > 0:
            job_count = len(lead.job_postings)
            weight = min(15, 5 + job_count)  # Cap at 15
            positive.append(
                ExtractedSignal(
                    tag="HIRING",
                    category="growth_indicators",
                    weight=weight,
                    matched_text=f"{job_count} job postings found",
                )
            )

        # Technology signals
        if lead.technologies and len(lead.technologies) > 0:
            tech_count = len(lead.technologies)
            if tech_count >= 5:
                positive.append(
                    ExtractedSignal(
                        tag="TECH_MATURE",
                        category="fit_indicators",
                        weight=8,
                        matched_text=f"Tech stack: {', '.join(lead.technologies[:5])}",
                    )
                )

        # Company age signals
        if lead.founded_year:
            import datetime
            current_year = datetime.datetime.now().year
            age = current_year - lead.founded_year

            if age < 2:
                # Very new company - could be risky
                negative.append(
                    ExtractedSignal(
                        tag="VERY_NEW",
                        category="red_flags",
                        weight=-5,
                        matched_text=f"Founded {lead.founded_year} ({age} years old)",
                    )
                )
            elif 2 <= age <= 10:
                # Sweet spot - established but still growing
                positive.append(
                    ExtractedSignal(
                        tag="GROWTH_PHASE",
                        category="fit_indicators",
                        weight=6,
                        matched_text=f"Founded {lead.founded_year} ({age} years old)",
                    )
                )

        return {"positive": positive, "negative": negative}

    def _calculate_funding_weight(self, amount: float) -> int:
        """Calculate weight based on funding amount"""
        if amount >= 100_000_000:  # $100M+
            return 20
        elif amount >= 50_000_000:  # $50M+
            return 18
        elif amount >= 10_000_000:  # $10M+
            return 15
        elif amount >= 5_000_000:  # $5M+
            return 12
        elif amount >= 1_000_000:  # $1M+
            return 10
        else:
            return 5

    def _get_match_context(self, text: str, match: re.Match, window: int = 50) -> str:
        """Get surrounding context for a match"""
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)

        context = text[start:end].strip()

        # Add ellipsis if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return context

    def _deduplicate_signals(self, signals: List[ExtractedSignal]) -> List[ExtractedSignal]:
        """Remove duplicate signals, keeping highest weight for each tag"""
        seen = {}
        for signal in signals:
            if signal.tag not in seen:
                seen[signal.tag] = signal
            elif abs(signal.weight) > abs(seen[signal.tag].weight):
                seen[signal.tag] = signal

        return list(seen.values())

    def get_signal_summary(self, result: SignalResult) -> Dict[str, Any]:
        """Generate a human-readable summary of signals"""
        return {
            "positive_count": len(result.positive),
            "negative_count": len(result.negative),
            "net_score": result.net_signal_score,
            "top_positive": [
                {"tag": s.tag, "evidence": s.matched_text}
                for s in sorted(result.positive, key=lambda x: x.weight, reverse=True)[:3]
            ],
            "top_negative": [
                {"tag": s.tag, "evidence": s.matched_text}
                for s in sorted(result.negative, key=lambda x: x.weight)[:3]
            ],
        }
