"""
Configuration for EG-CFG (Execution-Guided Constrained Function Generation)
for Ballerina code generation.
"""

# ─── Model Configuration ───────────────────────────────────────────────
MODEL_NAME = "lora_model"  # Path to your fine-tuned LoRA model
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto-detection
LOAD_IN_4BIT = True

# ─── Decoding Configuration ────────────────────────────────────────────
# Beam search parameters
BEAM_WIDTH = 5  # Number of candidate continuations per step
MAX_NEW_TOKENS_PER_LINE = 128  # Max tokens to generate for a single line
MAX_TOTAL_LINES = 50  # Max lines in a generated function body
MAX_NEW_TOKENS_TOTAL = 512  # Absolute token budget for the whole function

# Temperature & sampling
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50

# Retry / fallback
MAX_LINE_RETRIES = 5  # How many times to retry a line before falling back
MAX_FULL_RETRIES = 3  # How many times to retry the entire generation

# ─── Sandbox Configuration ─────────────────────────────────────────────
SANDBOX_BASE_DIR = "/tmp/eg_cfg_sandbox"
BAL_COMMAND = "bal"
COMPILE_TIMEOUT = 60  # seconds
TEST_TIMEOUT = 120  # seconds

# ─── Prompt Template ───────────────────────────────────────────────────
# Alpaca-style prompt (adjust to match your fine-tuning format)
PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Complete the following Ballerina function. Only output the complete function, no explanation.

{prompt}

### Response:
"""

# ─── Output ────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
RESULTS_FILE = "results/eg_cfg_results.json"
LOG_FILE = "results/eg_cfg.log"
