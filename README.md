# EG-CFG: Execution-Guided Constrained Function Generation

Generate error-free **Ballerina** code using an LLM with **line-by-line compile-time validation**.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                     eg_cfg.py                          │
│                  (Orchestrator)                         │
│  Load model → Load problems → For each problem:        │
│    ┌──────────────────────────────────────────────┐     │
│    │              decoder.py                      │     │
│    │         (EG-CFG Decoder)                     │     │
│    │                                              │     │
│    │  ┌─── Line-by-Line Strategy ───────────┐    │     │
│    │  │                                      │    │     │
│    │  │  1. Generate N candidate next-lines  │    │     │
│    │  │     (beam search w/ temperature)     │    │     │
│    │  │              │                       │    │     │
│    │  │  2. For each candidate:              │    │     │
│    │  │     Append to code-so-far            │    │     │
│    │  │              │                       │    │     │
│    │  │  3. ┌──── sandbox.py ────────┐      │    │     │
│    │  │     │  bal build (compile)    │      │    │     │
│    │  │     │  ✓ pass → keep line    │      │    │     │
│    │  │     │  ✗ fail → discard line │      │    │     │
│    │  │     └────────────────────────┘      │    │     │
│    │  │              │                       │    │     │
│    │  │  4. Repeat until braces balanced     │    │     │
│    │  └──────────────────────────────────────┘    │     │
│    │              │                               │     │
│    │  5. Run full test suite: bal test             │     │
│    │     ✓ pass → save & next problem             │     │
│    │     ✗ fail → retry (up to MAX_FULL_RETRIES)  │     │
│    └──────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────┘
```

## How EG-CFG Works

**Traditional decoding** generates the entire function at once — if there's a syntax error on line 5, the model has no way to know until the whole thing is done.

**EG-CFG** is different. It mimics how a human writes code:

1. **Write a line** → The model generates `BEAM_WIDTH` (default 5) candidate next-lines using sampling with temperature diversity.

2. **Compile it** → Each candidate is appended to the code-so-far, and the Ballerina compiler (`bal build`) checks it in a sandbox.

3. **Keep or discard** → If it compiles, keep the line. If it causes `SyntaxError`, type errors, or any compilation failure — **discard those tokens and force the model to try a different path**.

4. **Repeat** → Continue line-by-line until the function's braces are balanced (function complete).

5. **Test** → Run the complete function against unit tests with `bal test`.

6. **Retry** → If tests fail, retry the entire generation with different random seeds.

### Fallback Strategy

If line-by-line decoding can't find a valid path (e.g., all candidates for a line fail), it falls back to **full generation mode** — generate the entire function at once, extract it, and compile-check.

## Files

| File | Purpose |
|------|---------|
| `eg_cfg.py` | Main orchestrator — load model, iterate problems, save results |
| `decoder.py` | EG-CFG decoder — line-by-line generation with compile filtering |
| `sandbox.py` | Ballerina sandbox — `bal build` and `bal test` in isolated projects |
| `utils.py` | Helpers — parse functions, count braces, extract code blocks |
| `config.py` | All tunable parameters (beam width, temps, timeouts, etc.) |
| `problems.json` | Input problems with prompts and test cases |

## Prerequisites

```bash
# Python packages
pip install unsloth torch transformers

# Ballerina
# https://ballerina.io/downloads/
bal version  # Should show 2201.13.1 or compatible
```

## Usage

### 1. Test the sandbox (no model needed)

```bash
python eg_cfg.py --dry-run
```

This verifies that Ballerina compilation and testing works correctly.

### 2. Run all problems

```bash
python eg_cfg.py
```

### 3. Run specific problems

```bash
python eg_cfg.py --problems 0,1,2
```

### 4. Verbose mode

```bash
python eg_cfg.py --verbose
```

## Configuration

Edit `config.py` to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `"lora_model"` | Path to your fine-tuned LoRA model |
| `BEAM_WIDTH` | `5` | Number of candidate lines per step |
| `MAX_FULL_RETRIES` | `3` | Retries for the entire function |
| `MAX_LINE_RETRIES` | `5` | Retries per line |
| `TEMPERATURE` | `0.7` | Base sampling temperature |
| `MAX_TOTAL_LINES` | `50` | Max lines in function body |
| `MAX_NEW_TOKENS_TOTAL` | `512` | Token budget per function |
| `COMPILE_TIMEOUT` | `60s` | Timeout for `bal build` |
| `TEST_TIMEOUT` | `120s` | Timeout for `bal test` |

### Prompt Template

The prompt template in `config.py` should match your fine-tuning format. Default is Alpaca-style:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Complete the following Ballerina function. Only output the complete function, no explanation.

{prompt}

### Response:
```

## Output

Results are saved to `results/eg_cfg_results.json`:

```json
{
  "timestamp": "2026-03-03T...",
  "summary": {
    "total": 29,
    "passed": 25,
    "pass_rate": 86.21
  },
  "results": [
    {
      "problem_id": "request-130",
      "success": true,
      "compile_passed": true,
      "tests_passed": true,
      "test_details": {"passing": 12, "failing": 0, "skipped": 0},
      "attempts": 1,
      "elapsed_seconds": 45.2,
      "code": "function sumSquares(float[] lst) returns int { ... }"
    }
  ]
}
```

Logs are written to `results/eg_cfg.log`.
