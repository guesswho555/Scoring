"""
Configuration settings for ICP Scoring Engine
"""

from typing import Dict, List, Any
import os

# =============================================================================
# LLM CONFIGURATION (OpenRouter)
# =============================================================================

LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "openrouter"),  # openrouter, openai, anthropic
    "model": os.getenv("LLM_MODEL", "openai/gpt-4-turbo"),  # OpenRouter model format
    "api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "base_url": os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
    "max_tokens": 1000,
    "temperature": 0.3,
    # OpenRouter specific headers
    "site_url": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000"),
    "app_name": os.getenv("OPENROUTER_APP_NAME", "ICP Scoring Engine"),
}

# =============================================================================
# DEFAULT SCORING WEIGHTS
# =============================================================================

DEFAULT_WEIGHTS = {
    "firmographic": 0.35,
    "technographic": 0.25,
    "behavioral": 0.20,
    "signal_boost": 0.20,
}

# =============================================================================
# DEFAULT THRESHOLDS
# =============================================================================

DEFAULT_THRESHOLDS = {
    "auto_qualify": 80,
    "review_min": 50,
    "review_max": 79,
    "auto_reject": 49,
    "llm_enrichment_threshold": 60,
}

# =============================================================================
# GRADE MAPPING
# =============================================================================

GRADE_MAPPING = [
    (90, 100, "A+", "Exceptional - Fast track"),
    (80, 89, "A", "Excellent - High priority"),
    (70, 79, "B", "Good - Standard process"),
    (60, 69, "C", "Moderate - Needs review"),
    (50, 59, "D", "Weak - Low priority"),
    (0, 49, "F", "Poor - Auto-reject"),
]

# =============================================================================
# DEFAULT HARD FILTERS
# =============================================================================

DEFAULT_HARD_FILTERS = {
    "stop_list": {
        "domains": [],
        "keywords": [
            "student project",
            "test company",
            "example corp",
            "demo account",
            "fake company",
        ],
        "industries": [
            "gambling",
            "tobacco",
            "adult entertainment",
        ],
    },
    "data_requirements": {
        "min_fields_populated": 3,
        "required_fields": ["company_name"],
    },
    "geographic_exclusions": [],
    "company_size_gate": {
        "enabled": False,
        "min_employees": 5,
    },
}

# =============================================================================
# SIGNAL LIBRARY
# =============================================================================

SIGNAL_LIBRARY = {
    "positive_signals": {
        "growth_indicators": [
            {"pattern": r"series\s*[a-d]", "weight": 15, "tag": "FUNDED"},
            {"pattern": r"raised\s*\$[\d.]+\s*[mkb]", "weight": 12, "tag": "FUNDED"},
            {"pattern": r"(hiring|we'?re growing|expanding)", "weight": 10, "tag": "GROWING"},
            {"pattern": r"(scaling|scale-up|scale up)", "weight": 10, "tag": "SCALING"},
            {"pattern": r"(ipo|going public)", "weight": 20, "tag": "IPO"},
            {"pattern": r"(acquired|acquisition)", "weight": 8, "tag": "M&A"},
        ],
        "need_indicators": [
            {"pattern": r"(looking for|searching for|need|seeking)", "weight": 8, "tag": "ACTIVE_NEED"},
            {"pattern": r"(legacy|outdated|replacing|moderniz)", "weight": 12, "tag": "REPLACEMENT"},
            {"pattern": r"(manual process|spreadsheet|excel)", "weight": 10, "tag": "PAIN_POINT"},
            {"pattern": r"(automat|streamlin|efficien)", "weight": 8, "tag": "OPTIMIZATION"},
            {"pattern": r"(challenge|struggle|problem)", "weight": 6, "tag": "PAIN_POINT"},
        ],
        "fit_indicators": [
            {"pattern": r"\bb2b\b", "weight": 8, "tag": "B2B_FIT"},
            {"pattern": r"(enterprise|saas|software)", "weight": 8, "tag": "TECH_FIT"},
            {"pattern": r"(api|integration|platform)", "weight": 6, "tag": "TECH_FIT"},
            {"pattern": r"(startup|tech company)", "weight": 5, "tag": "TECH_FIT"},
        ],
        "budget_indicators": [
            {"pattern": r"(budget|invest|spending)", "weight": 8, "tag": "BUDGET"},
            {"pattern": r"(procurement|vendor|solution)", "weight": 6, "tag": "BUYING"},
            {"pattern": r"(rfp|request for proposal)", "weight": 15, "tag": "ACTIVE_BUYING"},
        ],
    },
    "negative_signals": {
        "disqualifiers": [
            {"pattern": r"(not accepting|closed|shutting down)", "weight": -20, "tag": "CLOSED"},
            {"pattern": r"(non-?profit|non profit)", "weight": -5, "tag": "NONPROFIT"},
            {"pattern": r"(government|public sector|federal)", "weight": -3, "tag": "GOV"},
            {"pattern": r"(free|open source|oss)", "weight": -3, "tag": "FREE"},
        ],
        "red_flags": [
            {"pattern": r"(lawsuit|litigation|sued)", "weight": -10, "tag": "LEGAL_RISK"},
            {"pattern": r"(layoffs|downsizing|restructur)", "weight": -8, "tag": "CONTRACTION"},
            {"pattern": r"(bankruptcy|insolvent|chapter 11)", "weight": -15, "tag": "FINANCIAL_RISK"},
            {"pattern": r"(fraud|scam|investigation)", "weight": -20, "tag": "FRAUD_RISK"},
        ],
        "competitor_indicators": [
            {"pattern": r"(competitor|competing|rival)", "weight": -25, "tag": "COMPETITOR"},
        ],
    },
}

# =============================================================================
# INDUSTRY MAPPING
# =============================================================================

INDUSTRY_CATEGORIES = {
    "primary_tech": ["SaaS", "Software", "Technology", "IT Services", "Cloud Computing"],
    "fintech": ["FinTech", "Financial Services", "Banking", "Insurance", "Payments"],
    "healthtech": ["HealthTech", "Healthcare", "Medical Devices", "Biotech", "Pharma"],
    "ecommerce": ["E-commerce", "Retail", "Consumer Goods", "Marketplace"],
    "martech": ["MarTech", "Marketing", "Advertising", "Media", "Digital Marketing"],
    "enterprise": ["Enterprise Software", "B2B", "Business Services", "Consulting"],
}

# =============================================================================
# TECH STACK MAPPING
# =============================================================================

TECH_CATEGORIES = {
    "cloud": ["aws", "azure", "gcp", "google cloud", "cloud", "heroku", "digitalocean"],
    "languages": ["python", "javascript", "typescript", "java", "go", "rust", "ruby"],
    "frontend": ["react", "vue", "angular", "next.js", "svelte"],
    "backend": ["node", "django", "flask", "fastapi", "spring", "rails"],
    "data": ["postgresql", "mysql", "mongodb", "redis", "elasticsearch", "kafka"],
    "devops": ["docker", "kubernetes", "terraform", "jenkins", "github actions", "ci/cd"],
    "ai_ml": ["machine learning", "ai", "tensorflow", "pytorch", "openai", "llm"],
}
