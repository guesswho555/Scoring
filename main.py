"""
ICP Scoring Engine - Main Entry Point
======================================
Run this file to start the FastAPI server.

Usage:
    python main.py                    # Start server on port 8000
    python main.py --port 8080        # Start server on custom port
    python main.py --reload           # Start with auto-reload (dev mode)

API Documentation:
    http://localhost:8000/docs        # Swagger UI
    http://localhost:8000/redoc       # ReDoc
"""

import argparse
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="ICP Scoring Engine API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    args = parser.parse_args()

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   ICP SCORING ENGINE                         ║
    ║                      Version 1.0.0                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Server starting on http://{args.host}:{args.port}                    ║
    ║  API Docs: http://localhost:{args.port}/docs                       ║
    ║  Health:   http://localhost:{args.port}/api/health                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "icp_engine.api.endpoints:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
