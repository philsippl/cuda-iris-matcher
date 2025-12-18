#!/usr/bin/env python3
"""Generate API documentation automatically from docstrings.

Usage:
    python docs/generate.py          # Generate both HTML and Markdown
    python docs/generate.py --html   # Generate HTML only
    python docs/generate.py --md     # Generate Markdown only
    python docs/generate.py --serve  # Serve HTML docs locally with live reload
"""

import argparse
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


def generate_html(output_dir: Path, modules: list):
    """Generate HTML documentation using pdoc."""
    try:
        import pdoc
    except ImportError:
        print("Installing pdoc...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdoc", "-q"])

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pdoc",
        "--output-directory", str(output_dir),
        "-d", "google",
    ] + modules
    subprocess.run(cmd)
    
    html_file = output_dir / "cuda_iris_matcher.html"
    if html_file.exists():
        print(f"✓ HTML docs: {html_file}")
        return True
    return False


def _format_signature(func: Callable) -> str:
    """Get function signature as string."""
    try:
        sig = inspect.signature(func)
        return f"{func.__name__}{sig}"
    except (ValueError, TypeError):
        return func.__name__ + "(...)"


def _generate_function_md(name: str, func: Callable) -> str:
    """Generate markdown for a single function."""
    lines = [f"### `{name}`\n"]
    
    # Signature
    sig = _format_signature(func)
    lines.append(f"```python\n{sig}\n```\n")
    
    # Docstring
    doc = inspect.getdoc(func)
    if doc:
        lines.append(doc + "\n")
    
    return "\n".join(lines)


def _generate_class_md(name: str, cls: type) -> str:
    """Generate markdown for a class."""
    lines = [f"### `{name}`\n"]
    
    doc = inspect.getdoc(cls)
    if doc:
        lines.append(doc + "\n")
    
    # Document fields for dataclasses
    if hasattr(cls, "__dataclass_fields__"):
        lines.append("\n**Fields:**\n")
        for field_name, field in cls.__dataclass_fields__.items():
            field_type = getattr(field.type, "__name__", str(field.type))
            lines.append(f"- `{field_name}`: {field_type}")
        lines.append("")
    
    return "\n".join(lines)


def generate_markdown(output_dir: Path):
    """Generate Markdown documentation directly from module."""
    import cuda_iris_matcher as ih
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# cuda_iris_matcher API Reference\n",
        "Auto-generated API documentation.\n",
        "## Functions\n",
    ]
    
    # Get all exported items
    all_items = ih.__all__
    
    functions = []
    classes = []
    constants = []
    
    for name in all_items:
        obj = getattr(ih, name)
        if inspect.isfunction(obj):
            functions.append((name, obj))
        elif inspect.isclass(obj):
            classes.append((name, obj))
        else:
            constants.append((name, obj))
    
    # Document functions
    for name, func in functions:
        lines.append(_generate_function_md(name, func))
        lines.append("---\n")
    
    # Document classes
    if classes:
        lines.append("## Classes\n")
        for name, cls in classes:
            lines.append(_generate_class_md(name, cls))
            lines.append("---\n")
    
    # Document constants
    if constants:
        lines.append("## Constants\n")
        lines.append("| Name | Value | Description |\n")
        lines.append("|------|-------|-------------|\n")
        for name, val in constants:
            if name.startswith("CATEGORY_"):
                desc = "Classification category"
            elif name.startswith("INCLUDE_"):
                desc = "Include flag for filtering"
            elif name.startswith("DEFAULT_"):
                desc = "Default dimension"
            else:
                desc = ""
            lines.append(f"| `{name}` | `{val}` | {desc} |\n")
    
    content = "\n".join(lines)
    
    output_file = output_dir / "cuda_iris_matcher.md"
    output_file.write_text(content)
    print(f"✓ Markdown docs: {output_file}")
    return True


def serve_docs(modules: list, port: int):
    """Serve HTML documentation with live reload."""
    try:
        import pdoc
    except ImportError:
        print("Installing pdoc...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdoc", "-q"])

    cmd = [
        sys.executable, "-m", "pdoc",
        "--host", "0.0.0.0",
        "--port", str(port),
        "-d", "google",
    ] + modules
    print(f"Serving docs at http://localhost:{port}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Generate API documentation")
    parser.add_argument("--serve", action="store_true", help="Serve docs with live reload")
    parser.add_argument("--html", action="store_true", help="Generate HTML only")
    parser.add_argument("--md", action="store_true", help="Generate Markdown only")
    parser.add_argument("--output", "-o", default="docs/api", help="Output directory")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port for serve mode")
    args = parser.parse_args()

    modules = ["cuda_iris_matcher"]
    output_dir = Path(args.output)

    if args.serve:
        serve_docs(modules, args.port)
        return

    # Default: generate both if neither --html nor --md specified
    gen_html = args.html or (not args.html and not args.md)
    gen_md = args.md or (not args.html and not args.md)

    print(f"Generating docs to {output_dir}/\n")

    if gen_html:
        generate_html(output_dir, modules)
    
    if gen_md:
        generate_markdown(output_dir)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
