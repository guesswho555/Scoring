"""
Stage 1: Hard Filters
=====================
Fast rejection of leads that don't meet basic criteria.
This stage runs first to save processing costs on unqualified leads.

Filters:
- Stop-list (domains, keywords, industries)
- Data completeness
- Geographic exclusions
- Company size gate
"""

import time
from typing import Optional
from urllib.parse import urlparse

from ..models.schemas import LeadData, FilterResult
from ..models.icp_config import ICPConfig, HardFiltersConfig
from ..config.settings import DEFAULT_HARD_FILTERS


class HardFilterStage:
    """
    Stage 1: Apply hard filters to quickly reject unqualified leads.
    """

    def __init__(self, config: Optional[ICPConfig] = None):
        """
        Initialize with ICP configuration or use defaults.
        """
        if config:
            self.filters = config.hard_filters
        else:
            self.filters = HardFiltersConfig(**DEFAULT_HARD_FILTERS)

    def process(self, lead: LeadData) -> FilterResult:
        """
        Apply all hard filters to a lead.

        Args:
            lead: Lead data to filter

        Returns:
            FilterResult with pass/fail status and reason
        """
        start_time = time.time()
        checks_performed = 0

        # Check 1: Stop-list domains
        checks_performed += 1
        domain_result = self._check_domain_stoplist(lead)
        if not domain_result["passed"]:
            return self._create_result(
                passed=False,
                reason=domain_result["reason"],
                rejected_by="domain_stoplist",
                checks=checks_performed,
                start_time=start_time,
            )

        # Check 2: Stop-list keywords
        checks_performed += 1
        keyword_result = self._check_keyword_stoplist(lead)
        if not keyword_result["passed"]:
            return self._create_result(
                passed=False,
                reason=keyword_result["reason"],
                rejected_by="keyword_stoplist",
                checks=checks_performed,
                start_time=start_time,
            )

        # Check 3: Stop-list industries
        checks_performed += 1
        industry_result = self._check_industry_stoplist(lead)
        if not industry_result["passed"]:
            return self._create_result(
                passed=False,
                reason=industry_result["reason"],
                rejected_by="industry_stoplist",
                checks=checks_performed,
                start_time=start_time,
            )

        # Check 4: Data completeness
        checks_performed += 1
        data_result = self._check_data_completeness(lead)
        if not data_result["passed"]:
            return self._create_result(
                passed=False,
                reason=data_result["reason"],
                rejected_by="data_completeness",
                checks=checks_performed,
                start_time=start_time,
            )

        # Check 5: Geographic exclusions
        checks_performed += 1
        geo_result = self._check_geographic_exclusions(lead)
        if not geo_result["passed"]:
            return self._create_result(
                passed=False,
                reason=geo_result["reason"],
                rejected_by="geographic_exclusion",
                checks=checks_performed,
                start_time=start_time,
            )

        # Check 6: Company size gate
        checks_performed += 1
        size_result = self._check_company_size(lead)
        if not size_result["passed"]:
            return self._create_result(
                passed=False,
                reason=size_result["reason"],
                rejected_by="company_size_gate",
                checks=checks_performed,
                start_time=start_time,
            )

        # All checks passed
        return self._create_result(
            passed=True,
            reason=None,
            rejected_by=None,
            checks=checks_performed,
            start_time=start_time,
        )

    def _check_domain_stoplist(self, lead: LeadData) -> dict:
        """Check if lead's domain is in stop-list"""
        if not lead.url or not self.filters.stop_list.domains:
            return {"passed": True, "reason": None}

        try:
            domain = urlparse(lead.url).netloc.lower()
            # Remove www. prefix
            domain = domain.replace("www.", "")

            for blocked_domain in self.filters.stop_list.domains:
                blocked = blocked_domain.lower().replace("www.", "")
                if blocked in domain or domain in blocked:
                    return {
                        "passed": False,
                        "reason": f"Domain in stop-list: {blocked_domain}",
                    }
        except Exception:
            pass

        return {"passed": True, "reason": None}

    def _check_keyword_stoplist(self, lead: LeadData) -> dict:
        """Check if lead contains stop-list keywords"""
        if not self.filters.stop_list.keywords:
            return {"passed": True, "reason": None}

        # Combine searchable text
        searchable = self._get_searchable_text(lead).lower()

        for keyword in self.filters.stop_list.keywords:
            if keyword.lower() in searchable:
                return {
                    "passed": False,
                    "reason": f"Stop-word detected: '{keyword}'",
                }

        return {"passed": True, "reason": None}

    def _check_industry_stoplist(self, lead: LeadData) -> dict:
        """Check if lead's industry is in stop-list"""
        if not lead.industry or not self.filters.stop_list.industries:
            return {"passed": True, "reason": None}

        lead_industry = lead.industry.lower()
        for blocked_industry in self.filters.stop_list.industries:
            if blocked_industry.lower() in lead_industry:
                return {
                    "passed": False,
                    "reason": f"Industry in stop-list: {blocked_industry}",
                }

        return {"passed": True, "reason": None}

    def _check_data_completeness(self, lead: LeadData) -> dict:
        """Check if lead has minimum required data"""
        reqs = self.filters.data_requirements

        # Check required fields
        for field in reqs.required_fields:
            # Handle OR conditions (e.g., "industry OR description")
            if " OR " in field:
                fields = [f.strip() for f in field.split(" OR ")]
                has_any = any(
                    getattr(lead, f, None) for f in fields if hasattr(lead, f)
                )
                if not has_any:
                    return {
                        "passed": False,
                        "reason": f"Missing required field: {field}",
                    }
            else:
                value = getattr(lead, field, None)
                if not value:
                    return {
                        "passed": False,
                        "reason": f"Missing required field: {field}",
                    }

        # Check minimum populated fields
        populated = self._count_populated_fields(lead)
        if populated < reqs.min_fields_populated:
            return {
                "passed": False,
                "reason": f"Insufficient data: {populated}/{reqs.min_fields_populated} fields",
            }

        return {"passed": True, "reason": None}

    def _check_geographic_exclusions(self, lead: LeadData) -> dict:
        """Check if lead is in excluded geography"""
        if not self.filters.geographic_exclusions:
            return {"passed": True, "reason": None}

        # Check country
        if lead.country:
            for excluded in self.filters.geographic_exclusions:
                if excluded.lower() in lead.country.lower():
                    return {
                        "passed": False,
                        "reason": f"Geographic exclusion: {excluded}",
                    }

        # Check headquarters
        if lead.headquarters:
            for excluded in self.filters.geographic_exclusions:
                if excluded.lower() in lead.headquarters.lower():
                    return {
                        "passed": False,
                        "reason": f"Geographic exclusion: {excluded}",
                    }

        return {"passed": True, "reason": None}

    def _check_company_size(self, lead: LeadData) -> dict:
        """Check if company meets size requirements"""
        gate = self.filters.company_size_gate

        if not gate.enabled:
            return {"passed": True, "reason": None}

        if lead.employee_count is None:
            # If we don't have employee data, let it pass (can't filter)
            return {"passed": True, "reason": None}

        if lead.employee_count < gate.min_employees:
            return {
                "passed": False,
                "reason": f"Company too small: {lead.employee_count} < {gate.min_employees} employees",
            }

        if gate.max_employees and lead.employee_count > gate.max_employees:
            return {
                "passed": False,
                "reason": f"Company too large: {lead.employee_count} > {gate.max_employees} employees",
            }

        return {"passed": True, "reason": None}

    def _get_searchable_text(self, lead: LeadData) -> str:
        """Combine all searchable text fields"""
        parts = []

        if lead.company_name:
            parts.append(lead.company_name)
        if lead.description:
            parts.append(lead.description)
        if lead.raw_content:
            parts.append(lead.raw_content)
        if lead.meta_tags:
            parts.append(" ".join(lead.meta_tags))

        return " ".join(parts)

    def _count_populated_fields(self, lead: LeadData) -> int:
        """Count how many fields have data"""
        count = 0
        important_fields = [
            "company_name",
            "url",
            "industry",
            "description",
            "employee_count",
            "headquarters",
            "technologies",
            "funding_info",
        ]

        for field in important_fields:
            value = getattr(lead, field, None)
            if value:
                if isinstance(value, list) and len(value) > 0:
                    count += 1
                elif isinstance(value, dict) and len(value) > 0:
                    count += 1
                elif not isinstance(value, (list, dict)):
                    count += 1

        return count

    def _create_result(
        self,
        passed: bool,
        reason: Optional[str],
        rejected_by: Optional[str],
        checks: int,
        start_time: float,
    ) -> FilterResult:
        """Create a FilterResult object"""
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return FilterResult(
            passed=passed,
            checked_filters=checks,
            rejection_reason=reason,
            rejected_by=rejected_by,
            processing_time_ms=round(processing_time, 2),
        )
