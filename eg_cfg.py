#!/usr/bin/env python3
"""
EG-CFG: Execution-Guided Constrained Function Generation
=========================================================

Main orchestrator that:
1. Loads the fine-tuned Ballerina model via Unsloth
2. Reads problems from problems.json
3. For each problem, runs the EG-CFG decoder (line-by-line with
   compile-time sandbox filtering)
4. Validates against unit tests
5. Saves results to results/eg_cfg_results.json

Usage:
    python eg_cfg.py                        # Run all problems
    python eg_cfg.py --problems 0,1,2       # Run specific problem indices
    python eg_cfg.py --dry-run              # Test sandbox only, no model
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from unsloth import FastLanguageModel
from transformers import TextStreamer
import config
from sandbox import BallerinaSandbox
from decoder import EGCFGDecoder, DecodingResult
from utils import extract_function_name

# ─── Logging ───────────────────────────────────────────────────────
os.makedirs(config.RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("eg_cfg")


def load_model():
    """Load the fine-tuned model using Unsloth."""
    logger.info("Loading model from '%s' ...", config.MODEL_NAME)



    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        load_in_4bit=config.LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    logger.info("Model loaded successfully.")
    return model, tokenizer


def load_problems(path: str = "problems.json") -> list[dict]:
    """Load the problem set."""
    with open(path, "r") as f:
        problems = json.load(f)
    logger.info("Loaded %d problems from %s", len(problems), path)
    return problems


def run_dry_run(problems: list[dict]):
    """
    Test the sandbox infrastructure without loading the model.
    Verifies that Ballerina compilation and testing works.
    """
    logger.info("═══ DRY RUN — testing sandbox infrastructure ═══")

    sandbox = BallerinaSandbox("dry_run")

    # Test 1: Simple compile check
    test_code = 'function hello() returns string { return "hello"; }'
    result = sandbox.compile_check(test_code)
    logger.info("Compile check (valid code):   success=%s", result.success)
    assert result.success, f"Valid code should compile! Errors: {result.errors}"

    # Test 2: Syntax error should fail
    bad_code = "function broken() returns int { int x = }"
    result = sandbox.compile_check(bad_code)
    logger.info("Compile check (syntax error): success=%s", result.success)
    assert not result.success, "Syntax errors should fail compilation!"

    # Test 3: Full test run with a known-good solution
    good_code = (
        "function isEqualToSumEven(int n) returns boolean {\n"
        "    return n >= 8 && n % 2 == 0;\n"
        "}"
    )
    test_code_str = problems[4]["test"]  # isEqualToSumEven tests
    result = sandbox.test_check(good_code, test_code_str)
    logger.info("Test check (correct solution): success=%s", result.success)

    sandbox.cleanup()
    logger.info("═══ DRY RUN COMPLETE — sandbox is working! ═══")


def run_eg_cfg(
    problems: list[dict],
    problem_indices: list[int] | None = None,
):
    """
    Run the full EG-CFG pipeline on selected (or all) problems.
    """
    model, tokenizer = load_model()
    sandbox = BallerinaSandbox("main")

    decoder = EGCFGDecoder(model, tokenizer, sandbox)

    results: list[dict] = []
    total = 0
    passed = 0

    if problem_indices is None:
        problem_indices = list(range(len(problems)))

    for idx in problem_indices:
        prob = problems[idx]
        problem_id = prob["id"]
        prompt = prob["prompt"]
        test_code = prob["test"]

        logger.info(
            "\n╔══════════════════════════════════════════════════╗\n"
            "║  Problem %d/%d: %s (fn: %s)\n"
            "╚══════════════════════════════════════════════════╝",
            idx + 1, len(problems), problem_id,
            extract_function_name(prompt),
        )

        start_time = time.time()

        # Reset sandbox for each problem
        sandbox.reset()

        result: DecodingResult = decoder.generate(problem_id, prompt, test_code)

        elapsed = time.time() - start_time
        total += 1
        if result.success:
            passed += 1

        result_entry = {
            "index": idx,
            "problem_id": result.problem_id,
            "success": result.success,
            "compile_passed": result.compile_passed,
            "tests_passed": result.tests_passed,
            "test_details": result.test_details,
            "attempts": result.attempts,
            "compile_checks": result.compile_checks,
            "elapsed_seconds": round(elapsed, 2),
            "code": result.code,
            "error_messages": result.error_messages,
        }
        results.append(result_entry)

        status = "✅ PASS" if result.success else "❌ FAIL"
        logger.info(
            "%s — %s — attempts=%d, compile_checks=%d, time=%.1fs",
            status, problem_id, result.attempts, result.compile_checks, elapsed,
        )

        # Save intermediate results after each problem
        _save_results(results, passed, total)

    sandbox.cleanup()

    logger.info(
        "\n══════════════════════════════════════════════════\n"
        "  FINAL RESULTS: %d / %d passed (%.1f%%)\n"
        "══════════════════════════════════════════════════",
        passed, total, (passed / total * 100) if total > 0 else 0,
    )

    return results


def _save_results(results: list[dict], passed: int, total: int):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total * 100, 2) if total > 0 else 0,
        },
        "results": results,
    }
    with open(config.RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    logger.debug("Results saved to %s", config.RESULTS_FILE)


def main():
    parser = argparse.ArgumentParser(
        description="EG-CFG: Execution-Guided Constrained Function Generation for Ballerina"
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="Comma-separated list of problem indices to run (0-based). Default: all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test the sandbox infrastructure without loading the model",
    )
    parser.add_argument(
        "--problems-file",
        type=str,
        default="problems.json",
        help="Path to the problems JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    problems = load_problems(args.problems_file)

    if args.dry_run:
        run_dry_run(problems)
        return

    # Parse problem indices
    indices = None
    if args.problems:
        indices = [int(x.strip()) for x in args.problems.split(",")]
        logger.info("Running problems: %s", indices)

    run_eg_cfg(problems, problem_indices=indices)


if __name__ == "__main__":
    main()
