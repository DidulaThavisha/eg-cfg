"""
Execution-Guided Constrained Function Generation (EG-CFG) Decoder

Revised strategy — "Generate-then-Validate-then-Repair":

Phase 1 — Full Generation (fast path):
  Generate the entire function at once with multiple candidates.
  If any candidate compiles AND passes tests → done.

Phase 2 — Chunked Line-by-Line with Stub Validation:
  Generate the function body in chunks of lines.  After each chunk,
  compile-check by wrapping the partial body with a type-appropriate
  stub return + '}'.  Discard chunks that introduce new errors.

Phase 3 — Retry with different seeds.
"""

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
    extract_return_type,
    is_function_complete,
    strip_markdown_fences,
    extract_function_block,
    make_compilable_stub,
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
    compile_checks: int = 0


class EGCFGDecoder:
    """
    Execution-Guided decoder with multiple strategies.
    """

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, sandbox: BallerinaSandbox):
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox = sandbox
        self.device = next(model.parameters()).device
        self.compile_checks = 0

    # ─── Public API ─────────────────────────────────────────────────

    def generate(self, problem_id: str, prompt: str, test_code: str) -> DecodingResult:
        """
        Main entry point.  Tries strategies in order:
        1. Full generation with multiple candidates (fastest)
        2. Chunked line-by-line EG-CFG (most robust)
        """
        logger.info("═══ Generating solution for %s ═══", problem_id)
        self.compile_checks = 0
        best_code = ""
        best_compile_passed = False

        for attempt in range(1, config.MAX_FULL_RETRIES + 1):
            logger.info("Attempt %d/%d for %s", attempt, config.MAX_FULL_RETRIES, problem_id)

            # --- Strategy 1: Full generation (try multiple candidates) ---
            candidates = self._generate_full_candidates(prompt, seed_offset=attempt)
            for i, code in enumerate(candidates):
                if not code:
                    continue
                res = self._compile_check(code)
                if res.success:
                    best_code = code
                    best_compile_passed = True
                    logger.info("Full-gen candidate %d compiled ✓", i)
                    test_res = self.sandbox.test_check(code, test_code)
                    test_counts = count_tests(test_res.output)
                    if test_res.success:
                        logger.info("✅ All tests passed for %s (full-gen, attempt %d)!", problem_id, attempt)
                        return DecodingResult(
                            problem_id=problem_id, success=True, code=code,
                            compile_passed=True, tests_passed=True,
                            test_details=test_counts, attempts=attempt,
                            compile_checks=self.compile_checks,
                        )
                    else:
                        logger.info(
                            "Full-gen candidate %d: compile ✓ tests ✗ %s",
                            i, test_counts,
                        )
                else:
                    logger.debug("Full-gen candidate %d: compile ✗ %s", i, res.errors[:2])
                    if not best_code:
                        best_code = code

            # --- Strategy 2: Chunked line-by-line EG-CFG ---
            code = self._chunked_line_by_line_decode(prompt, seed_offset=attempt)
            if code:
                best_code = code
                res = self._compile_check(code)
                if res.success:
                    best_compile_passed = True
                    logger.info("Chunked EG-CFG compiled ✓ for %s", problem_id)
                    test_res = self.sandbox.test_check(code, test_code)
                    test_counts = count_tests(test_res.output)
                    if test_res.success:
                        logger.info("✅ All tests passed for %s (chunked EG-CFG)!", problem_id)
                        return DecodingResult(
                            problem_id=problem_id, success=True, code=code,
                            compile_passed=True, tests_passed=True,
                            test_details=test_counts, attempts=attempt,
                            compile_checks=self.compile_checks,
                        )
                    else:
                        logger.info("Chunked EG-CFG: compile ✓ tests ✗ %s", test_counts)

            self.sandbox.reset()

        # Exhausted all retries
        return DecodingResult(
            problem_id=problem_id, success=False, code=best_code,
            compile_passed=best_compile_passed, tests_passed=False,
            attempts=config.MAX_FULL_RETRIES,
            error_messages=["Exhausted all retries"],
            compile_checks=self.compile_checks,
        )

    # ─── Compile check wrapper (with counting) ─────────────────────

    def _compile_check(self, code: str) -> SandboxResult:
        """Compile-check with counter for profiling."""
        self.compile_checks += 1
        return self.sandbox.compile_check(code)

    # ─── Strategy 1: Full generation ───────────────────────────────

    def _generate_full_candidates(self, prompt: str, seed_offset: int = 0) -> list[str]:
        """
        Generate the entire function at once, multiple times with
        different temperatures.  Return list of extracted code strings.
        """
        model_input = self._build_model_input(prompt)
        input_ids = self._tokenize(model_input)

        logger.debug("Model input (first 300 chars): %s", model_input[:300])

        candidates = []

        for i in range(config.BEAM_WIDTH):
            torch.manual_seed(42 + seed_offset * 1000 + i * 7)

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
                    attention_mask=torch.ones_like(input_ids),
                )

            new_tokens = outputs[0][input_ids.shape[1]:]
            raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            logger.info("Full-gen raw output [%d] (%d chars):\n%.500s", i, len(raw_text), raw_text)

            # Try to extract a clean function block
            code = extract_function_block(raw_text)
            if not code:
                code = strip_markdown_fences(raw_text)
            if code:
                candidates.append(code)
            else:
                logger.debug("Full-gen candidate %d: could not extract code", i)

        logger.info("Generated %d full candidates", len(candidates))
        return candidates

    # ─── Strategy 2: Chunked line-by-line EG-CFG ──────────────────

    def _chunked_line_by_line_decode(
        self, prompt: str, seed_offset: int = 0,
        chunk_size: int = 3,
    ) -> Optional[str]:
        """
        Generate code in chunks of `chunk_size` lines.  After each chunk,
        wrap the partial body into a compilable stub and compile-check.
        Reject chunks that introduce new compile errors.

        This is much more practical than pure single-line because:
        - A chunk of 3 lines is more likely to form a complete statement
        - We use a type-appropriate stub return so partial functions compile
        - Far fewer compile checks (~10-15 instead of 40+)
        """
        signature = extract_function_signature(prompt)
        if not signature:
            logger.error("Could not extract function signature from prompt")
            return None

        return_type = extract_return_type(signature)
        imports = self._detect_imports(prompt)
        body_lines: list[str] = []

        model_input_base = self._build_model_input(prompt)

        for chunk_idx in range(config.MAX_TOTAL_LINES // chunk_size + 1):
            # Generate candidate chunks
            candidates = self._generate_candidate_chunks(
                model_input_base, body_lines, signature, imports,
                chunk_size=chunk_size,
                num_candidates=config.BEAM_WIDTH,
                seed_offset=seed_offset * 100 + chunk_idx,
            )

            if not candidates:
                logger.warning("No candidates generated at chunk %d", chunk_idx)
                break

            found_valid = False
            for cand_lines, cand_tokens in candidates:
                test_body = body_lines + cand_lines
                test_code = build_partial_code(imports, signature, test_body)

                # Check if the function is already complete (braces balanced)
                if is_function_complete(test_code):
                    result = self._compile_check(test_code)
                    if result.success:
                        logger.info("Chunk %d accepted (function complete, %d lines)",
                                    chunk_idx, len(test_body))
                        return test_code
                    else:
                        logger.debug("Chunk %d rejected (complete but errors): %s",
                                     chunk_idx, result.errors[:2])
                        continue

                # Not complete yet — wrap with a stub return to make it compilable
                stub_code = make_compilable_stub(imports, signature, test_body, return_type)
                result = self._compile_check(stub_code)
                if result.success:
                    body_lines = test_body
                    found_valid = True
                    logger.debug("Chunk %d accepted (%d body lines so far)",
                                 chunk_idx, len(body_lines))
                    break
                else:
                    logger.debug("Chunk %d rejected: %s", chunk_idx, result.errors[:2])

            if not found_valid:
                logger.warning("No valid chunk at position %d, stopping chunked decode", chunk_idx)
                break

        # Try to close the function with a closing brace
        if body_lines:
            full_code = build_partial_code(imports, signature, body_lines, close=True)
            if is_function_complete(full_code):
                result = self._compile_check(full_code)
                if result.success:
                    return full_code

        return None

    def _generate_candidate_chunks(
        self,
        model_input_base: str,
        existing_body: list[str],
        signature: str,
        imports: list[str],
        chunk_size: int,
        num_candidates: int,
        seed_offset: int,
    ) -> list[tuple[list[str], int]]:
        """
        Generate multiple candidate chunks (groups of lines) from the model.
        Returns list of (lines, token_count) tuples.
        """
        # Build context: prompt + imports + signature + body so far
        code_so_far_lines = list(imports)
        if imports:
            code_so_far_lines.append("")
        code_so_far_lines.append(signature)
        code_so_far_lines.extend(existing_body)

        full_input = model_input_base + "\n".join(code_so_far_lines) + "\n"
        input_ids = self._tokenize(full_input)

        candidates = []
        seen = set()

        for i in range(num_candidates):
            torch.manual_seed(42 + seed_offset * 13 + i * 7)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=config.MAX_NEW_TOKENS_PER_LINE * chunk_size,
                    do_sample=True,
                    temperature=config.TEMPERATURE + (i * 0.05),
                    top_p=config.TOP_P,
                    top_k=config.TOP_K,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(input_ids),
                )

            new_tokens = outputs[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Extract the next `chunk_size` lines
            all_lines = generated_text.split("\n")
            chunk_lines = []
            for line in all_lines:
                chunk_lines.append(line)
                if len(chunk_lines) >= chunk_size:
                    break
                # Stop early if we see a closing brace that would end the function
                if line.strip() == "}":
                    break

            if not chunk_lines:
                continue

            # Deduplicate by normalized content
            key = "\n".join(l.strip() for l in chunk_lines)
            if key in seen:
                continue
            seen.add(key)

            num_tokens = len(self.tokenizer.encode("\n".join(chunk_lines)))
            candidates.append((chunk_lines, num_tokens))

        return candidates

    # ─── Helpers ───────────────────────────────────────────────────

    def _build_model_input(self, prompt: str) -> str:
        """Build the model input using the chat template or prompt template."""
        # Try the tokenizer's chat template first (matches fine-tuning format)
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": (
                    "Complete the following Ballerina function. "
                    "Only output the complete function implementation, no explanation.\n\n"
                    f"{prompt}"
                )},
            ]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                logger.debug("Using chat template for model input")
                return formatted
            except Exception as e:
                logger.warning("Chat template failed (%s), falling back to PROMPT_TEMPLATE", e)

        # Fallback to config template
        return config.PROMPT_TEMPLATE.format(prompt=prompt)

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and move to device."""
        return self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
        ).input_ids.to(self.device)

    @staticmethod
    def _detect_imports(prompt: str) -> list[str]:
        """Detect if the prompt requires any Ballerina imports."""
        imports = []
        prompt_lower = prompt.lower()

        if "md5" in prompt_lower or "hash" in prompt_lower:
            imports.append("import ballerina/crypto;")
        if any(kw in prompt_lower for kw in ["ceiling", "ceil", "round"]):
            imports.append("import ballerina/lang.'float as floats;")
        if "math" in prompt_lower:
            imports.append("import ballerina/lang.'float as floats;")
        if "io:println" in prompt:
            imports.append("import ballerina/io;")
        if "regex" in prompt_lower or "re:" in prompt:
            imports.append("import ballerina/lang.regexp;")

        return imports
