#!/usr/bin/env python3
"""Generate API documentation automatically from docstrings.

Usage:
    python docs/generate.py          # Generate HTML docs to docs/api/
    python docs/generate.py --serve  # Serve docs locally with live reload
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate API documentation")
    parser.add_argument("--serve", action="store_true", help="Serve docs with live reload")
    parser.add_argument("--output", "-o", default="docs/api", help="Output directory")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port for serve mode")
    args = parser.parse_args()

    # Ensure pdoc is installed
    try:
        import pdoc
    except ImportError:
        print("Installing pdoc...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdoc", "-q"])

    modules = ["cuda_iris_matcher"]

    if args.serve:
        # Live server mode
        cmd = [
            sys.executable, "-m", "pdoc",
            "--host", "0.0.0.0",
            "--port", str(args.port),
            "-d", "google",
        ] + modules
        print(f"Serving docs at http://localhost:{args.port}")
        subprocess.run(cmd)
    else:
        # HTML output
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "pdoc",
            "--output-directory", str(output_dir),
            "-d", "google",
        ] + modules
        print(f"Generating HTML docs to {output_dir}/")
        subprocess.run(cmd)
        
        # Find the generated file
        html_file = output_dir / "cuda_iris_matcher.html"
        if html_file.exists():
            print(f"\n✓ Documentation generated: {html_file}")
            print(f"  Open in browser: file://{html_file.absolute()}")
        else:
            # Check for index
            index_file = output_dir / "index.html"
            if index_file.exists():
                print(f"\n✓ Documentation generated: {index_file}")


if __name__ == "__main__":
    main()
