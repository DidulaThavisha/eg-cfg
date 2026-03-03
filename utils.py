"""
Utility helpers for EG-CFG.
"""

import re
import logging

logger = logging.getLogger("eg_cfg.utils")


def extract_function_signature(prompt: str) -> str:
    """
    Extract the function signature line from a problem prompt.
    e.g. "function sumSquares(float[] lst) returns int {"
    """
    for line in prompt.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("function ") and "{" in stripped:
            return stripped
    return ""


def extract_function_name(prompt: str) -> str:
    """Extract the function name from a prompt."""
    sig = extract_function_signature(prompt)
    match = re.search(r"function\s+(\w+)\s*\(", sig)
    return match.group(1) if match else ""


def is_function_complete(code: str) -> bool:
    """
    Check if the generated code has balanced braces, meaning
    the function body is complete.
    """
    open_braces = code.count("{")
    close_braces = code.count("}")
    return open_braces > 0 and open_braces == close_braces


def strip_markdown_fences(text: str) -> str:
    """Remove ```ballerina ... ``` or ``` ... ``` fences."""
    text = text.strip()
    text = re.sub(r"^```(?:ballerina)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def extract_function_block(text: str) -> str:
    """
    Given model output that may contain commentary, extract just the
    function definition (from 'function ...' to its closing '}').
    Handles imports before the function as well.
    """
    text = strip_markdown_fences(text)
    lines = text.splitlines()
    result_lines = []
    in_function = False
    brace_count = 0

    for line in lines:
        stripped = line.strip()

        # Capture import statements
        if stripped.startswith("import ") and not in_function:
            result_lines.append(line)
            continue

        # Start capturing at function keyword
        if stripped.startswith("function ") and not in_function:
            in_function = True

        if in_function:
            result_lines.append(line)
            brace_count += line.count("{") - line.count("}")
            if brace_count <= 0 and brace_count != 0:
                break
            if brace_count == 0 and len(result_lines) > 1:
                break

    return "\n".join(result_lines)


def build_partial_code(
    imports: list[str],
    signature: str,
    body_lines: list[str],
    close: bool = False,
) -> str:
    """
    Assemble a partial (or complete) Ballerina source file from parts.
    """
    parts = []
    for imp in imports:
        parts.append(imp)
    if imports:
        parts.append("")
    parts.append(signature)
    for bl in body_lines:
        parts.append(bl)
    if close:
        parts.append("}")
    return "\n".join(parts)


def count_tests(test_output: str) -> dict:
    """
    Parse bal test output to count passing/failing/skipped.
    Example output line: '        3 passing'
    """
    counts = {"passing": 0, "failing": 0, "skipped": 0}
    for line in test_output.splitlines():
        stripped = line.strip()
        for key in counts:
            match = re.match(rf"(\d+)\s+{key}", stripped)
            if match:
                counts[key] = int(match.group(1))
    return counts
