"""
Stage 4: LLM Intelligence
=========================
AI-powered analysis for qualified leads only.
This stage is the most expensive, so it only runs for leads above the threshold.

Tasks:
- Relationship classification (Buyer/Partner/Competitor/No_Fit)
- Match explanation generation
- Talking points for sales
- Risk identification
"""

import time
import json
from typing import Optional, Dict, Any, List
import os

from ..models.schemas import (
    LeadData,
    SignalResult,
    ScoringResult,
    LLMResult,
    Risk,
    RelationshipType,
    Severity,
    UserCompanyProfile,
)
from ..config.settings import LLM_CONFIG


class LLMIntelligenceStage:
    """
    Stage 4: Generate intelligent analysis using LLM.
    Only runs for leads that pass the score threshold.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "openrouter"):
        """
        Initialize LLM client.

        Args:
            api_key: API key for LLM provider
            provider: LLM provider ("openrouter", "openai", or "anthropic")
        """
        self.api_key = api_key or LLM_CONFIG.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        self.provider = provider or LLM_CONFIG.get("provider", "openrouter")
        self.model = LLM_CONFIG.get("model", "openai/gpt-4-turbo")
        self.base_url = LLM_CONFIG.get("base_url", "https://openrouter.ai/api/v1")
        self.site_url = LLM_CONFIG.get("site_url", "http://localhost:8000")
        self.app_name = LLM_CONFIG.get("app_name", "ICP Scoring Engine")
        self.client = None

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on provider"""
        if not self.api_key:
            return

        try:
            if self.provider == "openrouter":
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    default_headers={
                        "HTTP-Referer": self.site_url,
                        "X-Title": self.app_name,
                    }
                )
            elif self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            elif self.provider == "anthropic":
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            pass

    def process(
        self,
        lead: LeadData,
        signals: SignalResult,
        scoring: ScoringResult,
        user_company: Optional[UserCompanyProfile] = None,
    ) -> LLMResult:
        """
        Generate LLM-powered analysis.

        Args:
            lead: Lead data
            signals: Signal extraction results from Stage 2
            scoring: Scoring results from Stage 3
            user_company: User's company profile

        Returns:
            LLMResult with relationship type, explanation, and recommendations
        """
        start_time = time.time()

        # If no LLM client, return a rule-based analysis
        if not self.client:
            return self._generate_rule_based_analysis(lead, signals, scoring, start_time)

        try:
            # Generate prompt
            prompt = self._generate_prompt(lead, signals, scoring, user_company)

            # Call LLM
            response = self._call_llm(prompt)

            # Parse response
            result = self._parse_response(response, start_time)

            return result

        except Exception as e:
            # Fallback to rule-based on error
            result = self._generate_rule_based_analysis(lead, signals, scoring, start_time)
            result.match_explanation += f" (LLM unavailable: {str(e)[:50]})"
            return result

    def _generate_prompt(
        self,
        lead: LeadData,
        signals: SignalResult,
        scoring: ScoringResult,
        user_company: Optional[UserCompanyProfile],
    ) -> str:
        """Generate the LLM prompt with context"""

        # User company context
        if user_company:
            user_context = f"""
USER COMPANY (The Seller):
- Name: {user_company.name}
- Core Offering: {user_company.core_offering}
- Value Proposition: {user_company.value_proposition or 'N/A'}
- Target Industries: {', '.join(user_company.target_industries) if user_company.target_industries else 'B2B Technology'}
"""
        else:
            user_context = """
USER COMPANY: B2B Technology Solution Provider
"""

        # Lead context
        lead_context = f"""
LEAD COMPANY (The Prospect):
- Name: {lead.company_name}
- Industry: {lead.industry or 'Unknown'}
- Description: {(lead.description or '')[:500]}
- Employee Count: {lead.employee_count or 'Unknown'}
- Location: {lead.headquarters or lead.country or 'Unknown'}
- Technologies: {', '.join(lead.technologies[:10]) if lead.technologies else 'Unknown'}
- Funding: {self._format_funding(lead.funding_info) if lead.funding_info else 'Unknown'}
"""

        # Pre-computed signals
        positive_tags = [s.tag for s in signals.positive[:5]]
        negative_tags = [s.tag for s in signals.negative[:3]]

        signals_context = f"""
PRE-COMPUTED ANALYSIS (Trust these signals):
- ICP Score: {scoring.overall_score}/100 (Grade: {scoring.grade.value})
- Qualification: {scoring.qualification_status.value}
- Positive Signals: {', '.join(positive_tags) if positive_tags else 'None'}
- Negative Signals: {', '.join(negative_tags) if negative_tags else 'None'}
- Firmographic Fit: {scoring.breakdown.firmographic.score}%
- Technographic Fit: {scoring.breakdown.technographic.score}%
- Behavioral Score: {scoring.breakdown.behavioral.score}%
"""

        prompt = f"""You are a B2B Sales Intelligence Analyst. Analyze the fit between the User Company and Lead Company.

{user_context}

{lead_context}

{signals_context}

TASK:
Provide your analysis as a valid JSON object with exactly these fields:

{{
  "relationship_type": "Buyer" | "Partner" | "Competitor" | "Investor" | "No_Fit",
  "relationship_confidence": 0-100,
  "match_explanation": "2-3 sentence explanation of why this is a good/bad fit",
  "talking_points": ["point 1", "point 2", "point 3", "point 4", "point 5"],
  "risks": [
    {{"risk": "description", "severity": "LOW" | "MEDIUM" | "HIGH", "mitigation": "suggestion"}}
  ],
  "recommended_action": "HIGH_PRIORITY_OUTREACH" | "STANDARD_OUTREACH" | "NURTURE" | "RESEARCH_MORE" | "DISQUALIFY",
  "suggested_channel": "LinkedIn/Email/Phone/etc",
  "urgency": "immediate/this_week/this_month/low"
}}

RULES:
1. If Competitor, relationship_confidence should be high and recommended_action should be DISQUALIFY
2. Talking points should be specific and reference actual data from the lead
3. Always include at least 1 risk, even for great fits
4. Base your analysis on the pre-computed signals - they are accurate

Return ONLY the JSON object, no other text."""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API"""
        if self.provider in ["openrouter", "openai"]:
            # Both OpenRouter and OpenAI use the same SDK interface
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a B2B sales intelligence analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.content[0].text

        raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, response: str, start_time: float) -> LLMResult:
        """Parse LLM response into structured result"""
        processing_time = (time.time() - start_time) * 1000

        try:
            # Clean response (remove markdown code blocks if present)
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            data = json.loads(clean)

            # Parse risks
            risks = []
            for r in data.get("risks", []):
                risks.append(Risk(
                    risk=r.get("risk", "Unknown risk"),
                    severity=Severity(r.get("severity", "MEDIUM")),
                    mitigation=r.get("mitigation"),
                ))

            return LLMResult(
                relationship_type=RelationshipType(data.get("relationship_type", "No_Fit")),
                relationship_confidence=data.get("relationship_confidence", 50),
                match_explanation=data.get("match_explanation", "Analysis unavailable"),
                talking_points=data.get("talking_points", [])[:5],
                risks=risks[:3],
                recommended_action=data.get("recommended_action", "RESEARCH_MORE"),
                suggested_channel=data.get("suggested_channel"),
                urgency=data.get("urgency"),
                llm_confidence=data.get("relationship_confidence", 50),
                processing_time_ms=round(processing_time, 2),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Return partial result on parse error
            return LLMResult(
                relationship_type=RelationshipType.NO_FIT,
                relationship_confidence=30,
                match_explanation=f"Analysis parsing failed: {str(e)[:100]}",
                talking_points=[],
                risks=[],
                recommended_action="RESEARCH_MORE",
                processing_time_ms=round(processing_time, 2),
            )

    def _generate_rule_based_analysis(
        self,
        lead: LeadData,
        signals: SignalResult,
        scoring: ScoringResult,
        start_time: float,
    ) -> LLMResult:
        """Generate analysis using rules when LLM is unavailable"""
        processing_time = (time.time() - start_time) * 1000

        # Determine relationship type based on signals
        competitor_signals = [s for s in signals.negative if s.tag == "COMPETITOR"]
        if competitor_signals:
            return LLMResult(
                relationship_type=RelationshipType.COMPETITOR,
                relationship_confidence=80,
                match_explanation=f"Potential competitor detected. {competitor_signals[0].matched_text or 'Review manually.'}",
                talking_points=[],
                risks=[Risk(risk="Direct competitor", severity=Severity.HIGH, mitigation="Verify competitive positioning")],
                recommended_action="DISQUALIFY",
                processing_time_ms=round(processing_time, 2),
            )

        # Determine based on score
        score = scoring.overall_score
        positive_tags = [s.tag for s in signals.positive]

        if score >= 80:
            relationship = RelationshipType.BUYER
            confidence = 85
            action = "HIGH_PRIORITY_OUTREACH"
            urgency = "this_week"
        elif score >= 60:
            relationship = RelationshipType.BUYER
            confidence = 70
            action = "STANDARD_OUTREACH"
            urgency = "this_month"
        elif score >= 40:
            relationship = RelationshipType.PARTNER if "TECH_FIT" in positive_tags else RelationshipType.BUYER
            confidence = 50
            action = "NURTURE"
            urgency = "low"
        else:
            relationship = RelationshipType.NO_FIT
            confidence = 60
            action = "DISQUALIFY"
            urgency = "low"

        # Generate explanation
        explanation = self._generate_rule_explanation(lead, signals, scoring)

        # Generate talking points
        talking_points = self._generate_rule_talking_points(lead, signals)

        # Generate risks
        risks = self._generate_rule_risks(lead, signals, scoring)

        return LLMResult(
            relationship_type=relationship,
            relationship_confidence=confidence,
            match_explanation=explanation,
            talking_points=talking_points,
            risks=risks,
            recommended_action=action,
            suggested_channel="LinkedIn + Email",
            urgency=urgency,
            llm_confidence=confidence,
            processing_time_ms=round(processing_time, 2),
        )

    def _generate_rule_explanation(
        self, lead: LeadData, signals: SignalResult, scoring: ScoringResult
    ) -> str:
        """Generate explanation using rules"""
        parts = []

        # Score summary
        grade = scoring.grade.value
        score = scoring.overall_score
        parts.append(f"{lead.company_name} scores {score}/100 (Grade {grade}).")

        # Top positive signals
        if signals.positive:
            top_signal = signals.positive[0]
            parts.append(f"Key signal: {top_signal.tag}.")

        # Firmographic fit
        firm_score = scoring.breakdown.firmographic.score
        if firm_score >= 70:
            if lead.industry:
                parts.append(f"Strong industry fit ({lead.industry}).")
        elif firm_score < 50:
            parts.append("Limited firmographic alignment.")

        return " ".join(parts)

    def _generate_rule_talking_points(
        self, lead: LeadData, signals: SignalResult
    ) -> List[str]:
        """Generate talking points using rules"""
        points = []

        # Funding talking point
        if lead.funding_info and lead.funding_info.total_raised:
            amount = lead.funding_info.total_raised
            if lead.funding_info.last_round:
                points.append(
                    f"Reference their {lead.funding_info.last_round} "
                    f"(${amount:,.0f}) - growth phase aligns with scaling needs"
                )

        # Hiring talking point
        if lead.job_postings and len(lead.job_postings) > 0:
            points.append(
                f"They're actively hiring ({len(lead.job_postings)} positions) - "
                "indicates growth and potential budget"
            )

        # Tech stack talking point
        if lead.technologies and len(lead.technologies) > 2:
            techs = ", ".join(lead.technologies[:3])
            points.append(f"Their tech stack ({techs}) suggests technical maturity")

        # Industry talking point
        if lead.industry:
            points.append(f"Experience serving {lead.industry} companies")

        # Growth signal talking point
        growth_signals = [s for s in signals.positive if s.tag in ["FUNDED", "GROWING", "SCALING"]]
        if growth_signals:
            points.append("Their growth trajectory indicates readiness for solutions like yours")

        return points[:5]

    def _generate_rule_risks(
        self, lead: LeadData, signals: SignalResult, scoring: ScoringResult
    ) -> List[Risk]:
        """Generate risks using rules"""
        risks = []

        # Check negative signals
        for signal in signals.negative[:2]:
            severity = Severity.HIGH if signal.weight <= -15 else Severity.MEDIUM
            risks.append(Risk(
                risk=f"{signal.tag}: {signal.matched_text[:50] if signal.matched_text else 'Detected'}",
                severity=severity,
                mitigation="Investigate before outreach",
            ))

        # Data completeness risk
        if scoring.data_completeness < 50:
            risks.append(Risk(
                risk="Limited data available for analysis",
                severity=Severity.LOW,
                mitigation="Research company further before outreach",
            ))

        # Size mismatch risk
        if lead.employee_count:
            if lead.employee_count > 5000:
                risks.append(Risk(
                    risk="Large enterprise - longer sales cycle expected",
                    severity=Severity.MEDIUM,
                    mitigation="Prepare enterprise-focused materials",
                ))
            elif lead.employee_count < 20:
                risks.append(Risk(
                    risk="Small company - limited budget likely",
                    severity=Severity.MEDIUM,
                    mitigation="Consider startup pricing tier",
                ))

        # Default risk if none found
        if not risks:
            risks.append(Risk(
                risk="Standard evaluation needed",
                severity=Severity.LOW,
                mitigation="Follow normal qualification process",
            ))

        return risks[:3]

    def _format_funding(self, funding_info) -> str:
        """Format funding info for prompt"""
        parts = []
        if funding_info.total_raised:
            parts.append(f"${funding_info.total_raised:,.0f} raised")
        if funding_info.last_round:
            parts.append(f"({funding_info.last_round})")
        return " ".join(parts) if parts else "Unknown"
