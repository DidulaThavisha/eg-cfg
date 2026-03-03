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


def extract_return_type(signature: str) -> str:
    """
    Extract the return type from a Ballerina function signature.
    e.g. "function foo(int x) returns int {" → "int"
         "function bar(string s) returns string[] {" → "string[]"
         "function baz(int x) returns [int?, int?] {" → "[int?, int?]"
         "function qux(int x) returns boolean {" → "boolean"
    Returns empty string if no return type found.
    """
    match = re.search(r"returns\s+(.+?)\s*\{", signature)
    if match:
        return match.group(1).strip()
    return ""


def make_compilable_stub(
    imports: list[str],
    signature: str,
    body_lines: list[str],
    return_type: str,
) -> str:
    """
    Wrap a partial function body with a type-appropriate stub return
    statement and closing brace, so it can be compile-checked even
    though the body is incomplete.

    For example, if the return type is 'int', appends:
        return 0;
    }
    """
    stub_return = _stub_return_for_type(return_type)
    parts = list(imports)
    if imports:
        parts.append("")
    parts.append(signature)
    parts.extend(body_lines)
    if stub_return:
        parts.append(f"    {stub_return}")
    parts.append("}")
    return "\n".join(parts)


def _stub_return_for_type(return_type: str) -> str:
    """
    Generate a dummy return statement for a given Ballerina return type.
    This allows partial functions to compile for validation.
    """
    rt = return_type.strip()
    if not rt:
        return "return;"

    # Nullable types (e.g. "string?")
    if rt.endswith("?"):
        return "return ();"

    # Tuple types (e.g. "[int?, int?]")
    if rt.startswith("[") and rt.endswith("]"):
        inner = rt[1:-1]
        parts = _split_tuple_types(inner)
        stubs = [_stub_value_for_type(p.strip()) for p in parts]
        return f"return [{', '.join(stubs)}];"

    # Array types (e.g. "int[]", "string[]")
    if rt.endswith("[]"):
        return "return [];"

    # Simple types
    return f"return {_stub_value_for_type(rt)};"


def _stub_value_for_type(t: str) -> str:
    """Return a default value expression for a Ballerina type."""
    t = t.strip()
    if t.endswith("?"):
        return "()"
    if t == "int":
        return "0"
    if t == "float" or t == "decimal":
        return "0.0"
    if t == "boolean":
        return "false"
    if t == "string":
        return '""'
    if t.endswith("[]"):
        return "[]"
    # Fallback
    return "0"


def _split_tuple_types(inner: str) -> list[str]:
    """
    Split comma-separated types inside a tuple, respecting nested brackets.
    e.g. "int?, int?" → ["int?", "int?"]
    """
    parts = []
    depth = 0
    current = ""
    for ch in inner:
        if ch in ("[", "("):
            depth += 1
        elif ch in ("]", ")"):
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


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
