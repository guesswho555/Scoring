#!/usr/bin/env python3
"""
ICP Scoring Engine - API Server
================================
Run this file to start the FastAPI server.

Usage:
    python run_server.py                # Start on port 8000
    python run_server.py --port 8080    # Start on custom port
    python run_server.py --reload       # Development mode with auto-reload

API Documentation:
    http://localhost:8000/docs          # Swagger UI
    http://localhost:8000/redoc         # ReDoc
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="ICP Scoring Engine API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    llm_status = "Enabled" if api_key and len(api_key) > 10 else "Disabled (no API key)"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   ICP SCORING ENGINE API                     ║
║                      Version 1.0.0                           ║
╠══════════════════════════════════════════════════════════════╣
║  Server:    http://{args.host}:{args.port}                            ║
║  Docs:      http://localhost:{args.port}/docs                       ║
║  Health:    http://localhost:{args.port}/api/health                 ║
║  LLM:       {llm_status:<43}║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    POST /api/icp/score/quick  - Fast scoring (no LLM)        ║
║    POST /api/icp/score        - Full scoring with LLM        ║
║    POST /api/icp/score/batch  - Batch processing             ║
║    POST /api/test             - Test with sample data        ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "icp_engine.api.endpoints:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
