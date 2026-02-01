"""
ICP Scoring Engine - Usage Examples
===================================
This file demonstrates how to use the ICP Scoring Engine
both programmatically and via the API.
"""

# =============================================================================
# EXAMPLE 1: Direct Engine Usage (Programmatic)
# =============================================================================

def example_direct_usage():
    """Use the engine directly in Python code"""
    from icp_engine.engine import ICPScoringEngine, create_engine
    from icp_engine.models.schemas import LeadData, UserCompanyProfile
    from icp_engine.models.icp_config import ICPConfig

    # Create a simple engine with default settings
    engine = ICPScoringEngine()

    # Or create with custom settings
    engine = create_engine(
        target_industries=["SaaS", "FinTech", "B2B"],
        target_tech=["AWS", "Python", "React"],
        target_regions=["US", "UK", "EU"],
    )

    # Define your company profile
    user_company = UserCompanyProfile(
        name="Acme Solutions",
        core_offering="API Integration Platform",
        value_proposition="We help B2B companies integrate their tools 10x faster",
        target_industries=["SaaS", "FinTech", "Enterprise"],
        technologies=["AWS", "Python", "GraphQL"],
    )

    # Create a lead to score
    lead = LeadData(
        company_name="TechStartup Inc",
        industry="SaaS",
        description=(
            "We are a fast-growing B2B SaaS company that helps enterprises "
            "manage their customer data. Recently raised $15M Series B and "
            "actively hiring engineers. We use AWS and Python for our backend."
        ),
        employee_count=120,
        headquarters="San Francisco, CA",
        technologies=["AWS", "Python", "React", "PostgreSQL", "Docker"],
        funding_info={
            "total_raised": 15000000,
            "last_round": "Series B",
        },
        job_postings=[
            "Senior Backend Engineer",
            "Integration Specialist",
            "Product Manager",
        ],
        social_links={
            "linkedin": "https://linkedin.com/company/techstartup",
            "twitter": "https://twitter.com/techstartup",
        },
    )

    # Score the lead
    print("=" * 60)
    print("SCORING LEAD: TechStartup Inc")
    print("=" * 60)

    result = engine.score_lead(lead, user_company, skip_llm=True)

    # Display results
    print(f"\nCompany: {result.company_name}")
    print(f"Overall Score: {result.scoring.overall_score}/100")
    print(f"Grade: {result.scoring.grade.value}")
    print(f"Status: {result.scoring.qualification_status.value}")

    print("\n--- Score Breakdown ---")
    breakdown = result.scoring.breakdown
    print(f"  Firmographic:  {breakdown.firmographic.score:.1f} (weighted: {breakdown.firmographic.weighted:.1f})")
    print(f"  Technographic: {breakdown.technographic.score:.1f} (weighted: {breakdown.technographic.weighted:.1f})")
    print(f"  Behavioral:    {breakdown.behavioral.score:.1f} (weighted: {breakdown.behavioral.weighted:.1f})")
    print(f"  Signal Boost:  {breakdown.signal_boost.score:.1f} (weighted: {breakdown.signal_boost.weighted:.1f})")

    print("\n--- Signals Detected ---")
    print(f"  Positive: {len(result.signals.positive)}")
    for sig in result.signals.positive[:3]:
        print(f"    + {sig.tag}: {sig.matched_text[:50] if sig.matched_text else 'N/A'}...")
    print(f"  Negative: {len(result.signals.negative)}")
    for sig in result.signals.negative[:3]:
        print(f"    - {sig.tag}: {sig.matched_text[:50] if sig.matched_text else 'N/A'}...")

    print("\n--- Summary ---")
    print(f"  {result.summary['one_liner']}")
    print(f"  Priority: {result.summary['priority_rank']}")
    print(f"  Next Step: {result.summary['next_step']}")

    print(f"\nProcessing Time: {result.total_processing_time_ms:.2f}ms")

    return result


# =============================================================================
# EXAMPLE 2: Batch Processing
# =============================================================================

def example_batch_processing():
    """Process multiple leads at once"""
    from icp_engine.engine import ICPScoringEngine
    from icp_engine.models.schemas import LeadData

    engine = ICPScoringEngine()

    # Create multiple leads
    leads = [
        LeadData(
            company_name="FastGrowth Co",
            industry="FinTech",
            description="B2B payment processing platform, Series A funded",
            employee_count=80,
            technologies=["AWS", "Node.js", "React"],
            funding_info={"total_raised": 8000000, "last_round": "Series A"},
        ),
        LeadData(
            company_name="LegacyCorp",
            industry="Manufacturing",
            description="Traditional manufacturing company, established 1970",
            employee_count=5000,
            technologies=["SAP", "Oracle"],
        ),
        LeadData(
            company_name="AIStartup",
            industry="AI/ML",
            description="We build AI tools for developers, hiring rapidly",
            employee_count=25,
            technologies=["Python", "TensorFlow", "AWS"],
            job_postings=["ML Engineer", "Backend Dev", "DevOps"],
        ),
        LeadData(
            company_name="ConsultingFirm",
            industry="Consulting",
            description="Management consulting for Fortune 500",
            employee_count=200,
        ),
        LeadData(
            company_name="TinyShop",
            industry="Retail",
            description="Small local retail store",
            employee_count=3,
        ),
    ]

    print("=" * 60)
    print("BATCH SCORING: 5 Leads")
    print("=" * 60)

    # Score all leads
    result = engine.score_batch(
        leads=leads,
        llm_for_top_n=2,  # Only run LLM for top 2
        min_score_threshold=40,
    )

    print(f"\nProcessed: {result.processed}")
    print(f"Qualified: {result.qualified}")
    print(f"High Priority: {result.high_priority}")
    print(f"Rejected: {result.rejected}")
    print(f"LLM Enriched: {result.llm_enriched}")
    print(f"Total Time: {result.processing_time_ms:.2f}ms")

    print("\n--- Ranked Results ---")
    for i, r in enumerate(result.results, 1):
        status_icon = "✓" if r.filter_status.passed else "✗"
        print(
            f"  {i}. [{status_icon}] {r.company_name}: "
            f"{r.scoring.overall_score}/100 ({r.scoring.grade.value}) - "
            f"{r.scoring.qualification_status.value}"
        )

    return result


# =============================================================================
# EXAMPLE 3: Custom ICP Configuration
# =============================================================================

def example_custom_config():
    """Create a custom ICP configuration"""
    from icp_engine.engine import ICPScoringEngine
    from icp_engine.models.icp_config import (
        ICPConfig,
        HardFiltersConfig,
        StopList,
        FirmographicCriteria,
        TargetIndustries,
        CompanySize,
        TechnographicCriteria,
        ScoringWeights,
        ScoringThresholds,
    )

    # Create custom configuration
    config = ICPConfig(
        name="Enterprise SaaS ICP",
        description="Ideal customer profile for enterprise SaaS sales",

        # Custom hard filters
        hard_filters=HardFiltersConfig(
            stop_list=StopList(
                domains=["competitor1.com", "competitor2.com"],
                keywords=["student", "demo", "test"],
                industries=["gambling", "tobacco"],
            ),
        ),

        # Custom firmographic criteria
        firmographic_criteria=FirmographicCriteria(
            target_industries=TargetIndustries(
                primary=["SaaS", "FinTech", "Enterprise Software"],
                secondary=["E-commerce", "HealthTech"],
                weight_primary=100,
                weight_secondary=70,
            ),
            company_size=CompanySize(
                ideal_min=100,
                ideal_max=1000,
                acceptable_min=50,
                acceptable_max=5000,
            ),
        ),

        # Custom technographic criteria
        technographic_criteria=TechnographicCriteria(
            required_tech=["cloud", "api"],
            preferred_tech=["aws", "kubernetes", "python", "react"],
            competing_tech=["competitor_product"],
        ),

        # Custom weights
        weights=ScoringWeights(
            firmographic=0.40,  # Increase firmographic importance
            technographic=0.30,  # Increase tech importance
            behavioral=0.15,
            signal_boost=0.15,
        ),

        # Custom thresholds
        thresholds=ScoringThresholds(
            auto_qualify=85,  # Higher bar for auto-qualify
            review_min=60,
            auto_reject=40,
            llm_enrichment_threshold=70,
        ),
    )

    # Create engine with custom config
    engine = ICPScoringEngine(icp_config=config)

    print("=" * 60)
    print("CUSTOM ICP CONFIGURATION")
    print("=" * 60)
    print(f"Config Name: {config.name}")
    print(f"Weights: {config.weights}")
    print(f"Target Industries: {config.firmographic_criteria.target_industries.primary}")
    print(f"Ideal Company Size: {config.firmographic_criteria.company_size.ideal_min}-{config.firmographic_criteria.company_size.ideal_max}")

    return engine


# =============================================================================
# EXAMPLE 4: API Usage with requests
# =============================================================================

def example_api_usage():
    """Use the API via HTTP requests"""
    import requests

    BASE_URL = "http://localhost:8000"

    print("=" * 60)
    print("API USAGE EXAMPLE")
    print("=" * 60)
    print("Make sure the server is running: python main.py")
    print()

    # Example: Score a lead via API
    payload = {
        "lead_data": {
            "company_name": "APITestCorp",
            "industry": "SaaS",
            "description": "B2B software company, Series B funded, hiring engineers",
            "employee_count": 150,
            "technologies": ["AWS", "Python", "React"],
            "funding_info": {
                "total_raised": 20000000,
                "last_round": "Series B"
            }
        },
        "options": {
            "skip_llm": True
        }
    }

    print("Request payload:")
    print(f"  POST {BASE_URL}/api/icp/score")
    print(f"  {payload}")

    # Uncomment to actually make the request:
    # response = requests.post(f"{BASE_URL}/api/icp/score", json=payload)
    # print(f"\nResponse: {response.json()}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ICP SCORING ENGINE - USAGE EXAMPLES")
    print("=" * 60 + "\n")

    # Run examples
    print("\n[Example 1: Direct Usage]")
    example_direct_usage()

    print("\n" + "-" * 60)
    print("\n[Example 2: Batch Processing]")
    example_batch_processing()

    print("\n" + "-" * 60)
    print("\n[Example 3: Custom Configuration]")
    example_custom_config()

    print("\n" + "-" * 60)
    print("\n[Example 4: API Usage]")
    example_api_usage()

    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
