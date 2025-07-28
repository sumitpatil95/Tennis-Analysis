#!/usr/bin/env python3
"""
Script to run the Tennis Action Recognition API server
"""

import argparse
import sys
import uvicorn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Tennis Action Recognition API')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind the server to')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--log-level', type=str, default='info',
                       choices=['critical', 'error', 'warning', 'info', 'debug'],
                       help='Log level')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes')
    
    return parser.parse_args()


def main():
    """Main function to run the API server"""
    args = parse_arguments()
    
    print(f"Starting Tennis Action Recognition API server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Log level: {args.log_level}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers if not args.reload else 1  # Workers > 1 not compatible with reload
    )


if __name__ == "__main__":
    main()