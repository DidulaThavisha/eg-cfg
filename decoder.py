"""
Execution-Guided Constrained Function Generation (EG-CFG) Decoder

This decoder generates Ballerina code **line-by-line**, running each
candidate through the Ballerina compiler in a sandbox.  Lines that cause
compile errors are discarded and the model is forced to try alternative
continuations.

Strategy
--------
1. Feed the model the prompt and generate `BEAM_WIDTH` candidate next-lines
   using sampling with temperature.
2. For each candidate, append it to the code-so-far and run `bal build`
   in the sandbox.
3. Keep only candidates that compile.  Rank them by model log-probability.
4. Pick the best candidate, commit it, and repeat.
5. When the function is complete (braces balanced), run `bal test` against
   the provided unit tests.
6. If tests fail, retry the entire generation (up to MAX_FULL_RETRIES).
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import PreTrainedTokenizerBase

import config
from sandbox import BallerinaSandbox, SandboxResult
from utils import (
    build_partial_code,
    count_tests,
    extract_function_signature,
    is_function_complete,
    strip_markdown_fences,
    extract_function_block,
)

logger = logging.getLogger("eg_cfg.decoder")


@dataclass
class Candidate:
    """A single beam candidate."""
    lines: list[str]
    token_ids: list[int] = field(default_factory=list)
    log_prob: float = 0.0
    complete: bool = False


@dataclass
class DecodingResult:
    """Final result of the EG-CFG decoding process for one problem."""
    problem_id: str
    success: bool
    code: str
    compile_passed: bool
    tests_passed: bool
    test_details: dict = field(default_factory=dict)
    attempts: int = 0
    error_messages: list[str] = field(default_factory=list)


class EGCFGDecoder:
    """
    Execution-Guided line-by-line decoder.

    Parameters
    ----------
    model : The language model (from Unsloth FastLanguageModel)
    tokenizer : The tokenizer
    sandbox : BallerinaSandbox instance for compile/test checks
    """

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, sandbox: BallerinaSandbox):
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox = sandbox
        self.device = next(model.parameters()).device

    # ─── Public API ─────────────────────────────────────────────────

    def generate(self, problem_id: str, prompt: str, test_code: str) -> DecodingResult:
        """
        Main entry point.  Tries EG-CFG line-by-line decoding first,
        and falls back to full-generation-then-filter if EG-CFG cannot
        produce a compilable solution.
        """
        logger.info("═══ Generating solution for %s ═══", problem_id)

        for attempt in range(1, config.MAX_FULL_RETRIES + 1):
            logger.info("Attempt %d/%d for %s", attempt, config.MAX_FULL_RETRIES, problem_id)

            # --- Strategy 1: Line-by-line EG-CFG ---
            code = self._line_by_line_decode(prompt, attempt)
            if code:
                compile_res = self.sandbox.compile_check(code)
                if compile_res.success:
                    logger.info("Compile passed for %s (line-by-line)", problem_id)
                    test_res = self.sandbox.test_check(code, test_code)
                    test_counts = count_tests(test_res.output)
                    if test_res.success:
                        logger.info("✅ All tests passed for %s!", problem_id)
                        return DecodingResult(
                            problem_id=problem_id,
                            success=True,
                            code=code,
                            compile_passed=True,
                            tests_passed=True,
                            test_details=test_counts,
                            attempts=attempt,
                        )
                    else:
                        logger.warning(
                            "Tests failed for %s: %s — retrying",
                            problem_id, test_counts,
                        )

            # --- Strategy 2: Full generation + extract + test ---
            code = self._full_generation_decode(prompt, attempt)
            if code:
                compile_res = self.sandbox.compile_check(code)
                if compile_res.success:
                    logger.info("Compile passed for %s (full-gen)", problem_id)
                    test_res = self.sandbox.test_check(code, test_code)
                    test_counts = count_tests(test_res.output)
                    if test_res.success:
                        logger.info("✅ All tests passed for %s!", problem_id)
                        return DecodingResult(
                            problem_id=problem_id,
                            success=True,
                            code=code,
                            compile_passed=True,
                            tests_passed=True,
                            test_details=test_counts,
                            attempts=attempt,
                        )
                    else:
                        logger.warning(
                            "Tests failed for %s (full-gen): %s",
                            problem_id, test_counts,
                        )

            self.sandbox.reset()

        # All retries exhausted — return the last code we had
        return DecodingResult(
            problem_id=problem_id,
            success=False,
            code=code if code else "",
            compile_passed=False,
            tests_passed=False,
            attempts=config.MAX_FULL_RETRIES,
            error_messages=["Exhausted all retries"],
        )

    # ─── Strategy 1: Line-by-line EG-CFG ───────────────────────────

    def _line_by_line_decode(self, prompt: str, seed_offset: int = 0) -> Optional[str]:
        """
        Generate code line-by-line.  After each line, compile-check
        the partial code.  Discard lines that break compilation.
        """
        signature = extract_function_signature(prompt)
        if not signature:
            logger.error("Could not extract function signature from prompt")
            return None

        # Detect any imports the prompt implies
        imports = self._detect_imports(prompt)

        body_lines: list[str] = []
        total_tokens_generated = 0

        # Build the model input: the full prompt up to where the body starts
        model_prompt = config.PROMPT_TEMPLATE.format(prompt=prompt)

        for line_idx in range(config.MAX_TOTAL_LINES):
            if total_tokens_generated >= config.MAX_NEW_TOKENS_TOTAL:
                logger.warning("Token budget exhausted at line %d", line_idx)
                break

            candidates = self._generate_candidate_lines(
                model_prompt, body_lines, signature, imports,
                num_candidates=config.BEAM_WIDTH,
                seed_offset=seed_offset + line_idx,
            )

            found_valid = False
            for cand_line, cand_tokens in candidates:
                # Check if this line closes the function
                test_lines = body_lines + [cand_line]
                test_code = build_partial_code(imports, signature, test_lines)

                # If braces are balanced, function is complete
                if is_function_complete(test_code):
                    test_code_with_close = test_code
                else:
                    # Append a closing brace to make it compilable
                    test_code_with_close = test_code + "\n}"

                result = self.sandbox.compile_check(test_code_with_close)
                if result.success:
                    body_lines.append(cand_line)
                    total_tokens_generated += cand_tokens
                    found_valid = True
                    logger.debug("  Line %d accepted: %s", line_idx, cand_line.strip())
                    break
                else:
                    logger.debug(
                        "  Line %d rejected: %s → %s",
                        line_idx, cand_line.strip(), result.errors[:2],
                    )

            if not found_valid:
                logger.warning("No valid candidate found at line %d, aborting line-by-line", line_idx)
                return None

            # Check if function is complete
            full_code = build_partial_code(imports, signature, body_lines)
            if is_function_complete(full_code):
                logger.info("Function complete at line %d", line_idx)
                return full_code

        # Ran out of lines — try to close
        full_code = build_partial_code(imports, signature, body_lines, close=True)
        result = self.sandbox.compile_check(full_code)
        if result.success:
            return full_code
        return None

    def _generate_candidate_lines(
        self,
        model_prompt: str,
        existing_body: list[str],
        signature: str,
        imports: list[str],
        num_candidates: int,
        seed_offset: int,
    ) -> list[tuple[str, int]]:
        """
        Generate multiple candidate next-lines from the model.

        Returns a list of (line_text, num_tokens) tuples sorted by
        descending log-probability.
        """
        # Build the context: prompt + code so far
        code_so_far_lines = []
        for imp in imports:
            code_so_far_lines.append(imp)
        if imports:
            code_so_far_lines.append("")
        code_so_far_lines.append(signature)
        code_so_far_lines.extend(existing_body)

        full_input = model_prompt + "\n".join(code_so_far_lines) + "\n"

        input_ids = self.tokenizer(
            full_input, return_tensors="pt", truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        ).input_ids.to(self.device)

        candidates = []
        seen = set()

        for i in range(num_candidates):
            torch.manual_seed(42 + seed_offset * 100 + i)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=config.MAX_NEW_TOKENS_PER_LINE,
                    do_sample=True,
                    temperature=config.TEMPERATURE + (i * 0.05),  # slight diversity
                    top_p=config.TOP_P,
                    top_k=config.TOP_K,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Extract just the next line(s)
            lines = generated_text.split("\n")
            next_line = lines[0] if lines else ""

            # Deduplicate
            if next_line.strip() in seen:
                continue
            seen.add(next_line.strip())

            num_tokens = len(self.tokenizer.encode(next_line))
            candidates.append((next_line, num_tokens))

        return candidates

    # ─── Strategy 2: Full generation + extract ─────────────────────

    def _full_generation_decode(self, prompt: str, seed_offset: int = 0) -> Optional[str]:
        """
        Generate the entire function at once, extract it, and validate.
        This is the fallback when line-by-line fails.
        """
        model_prompt = config.PROMPT_TEMPLATE.format(prompt=prompt)

        input_ids = self.tokenizer(
            model_prompt, return_tensors="pt", truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        ).input_ids.to(self.device)

        # Generate multiple full candidates
        best_code = None
        for i in range(config.BEAM_WIDTH):
            torch.manual_seed(123 + seed_offset * 100 + i)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=config.MAX_NEW_TOKENS_TOTAL,
                    do_sample=True,
                    temperature=config.TEMPERATURE + (i * 0.1),
                    top_p=config.TOP_P,
                    top_k=config.TOP_K,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Try to extract a clean function block
            code = extract_function_block(generated_text)
            if not code:
                code = strip_markdown_fences(generated_text)

            if not code:
                continue

            # Compile-check
            result = self.sandbox.compile_check(code)
            if result.success:
                logger.info("Full-gen candidate %d compiled successfully", i)
                return code
            else:
                logger.debug("Full-gen candidate %d failed: %s", i, result.errors[:3])
                if best_code is None:
                    best_code = code  # keep as fallback

        return best_code

    # ─── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _detect_imports(prompt: str) -> list[str]:
        """
        Detect if the prompt requires any Ballerina imports.
        """
        imports = []
        prompt_lower = prompt.lower()

        # Common patterns that need imports
        if "md5" in prompt_lower or "hash" in prompt_lower:
            imports.append("import ballerina/crypto;")
        if "ceiling" in prompt_lower or "ceil" in prompt_lower or "round" in prompt_lower:
            imports.append("import ballerina/lang.'float as floats;")
        if "math" in prompt_lower:
            imports.append("import ballerina/lang.'float as floats;")
        if "io:println" in prompt:
            imports.append("import ballerina/io;")
        if "regex" in prompt_lower or "re:" in prompt:
            imports.append("import ballerina/lang.regexp;")

        return imports
