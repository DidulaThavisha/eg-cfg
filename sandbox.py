"""
Ballerina Sandbox — compile & test Ballerina code in an isolated project.

Provides two levels of validation:
  1. compile_check(code)  — fast syntax/type check via `bal build`
  2. test_check(code, tests) — full test execution via `bal test`
"""

import os
import shutil
import subprocess
import tempfile
import logging
from dataclasses import dataclass
from typing import Optional

import config

logger = logging.getLogger("eg_cfg.sandbox")


@dataclass
class SandboxResult:
    """Result of a sandbox compilation or test run."""
    success: bool
    errors: list[str]
    output: str
    return_code: int


class BallerinaSandbox:
    """
    Manages a temporary Ballerina project for compile-checking and
    test-running generated code.
    """

    def __init__(self, sandbox_id: str = "default"):
        self.sandbox_id = sandbox_id
        self.base_dir = os.path.join(config.SANDBOX_BASE_DIR, sandbox_id)
        self._ensure_project()

    def _ensure_project(self):
        """Create a fresh Ballerina project in the sandbox directory."""
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)

        os.makedirs(self.base_dir, exist_ok=True)

        # Create Ballerina.toml
        toml_content = (
            "[package]\n"
            f'org = "egcfg"\n'
            f'name = "sandbox_{self.sandbox_id}"\n'
            f'version = "0.1.0"\n'
            f'distribution = "2201.13.1"\n\n'
            "[build-options]\n"
            "observabilityIncluded = true\n"
        )
        with open(os.path.join(self.base_dir, "Ballerina.toml"), "w") as f:
            f.write(toml_content)

        # Create empty main.bal
        with open(os.path.join(self.base_dir, "main.bal"), "w") as f:
            f.write("")

        # Create tests directory
        os.makedirs(os.path.join(self.base_dir, "tests"), exist_ok=True)

    def _write_code(self, code: str, test_code: Optional[str] = None):
        """Write the generated code and optional tests to the project."""
        with open(os.path.join(self.base_dir, "main.bal"), "w") as f:
            f.write(code)

        test_path = os.path.join(self.base_dir, "tests", "main_test.bal")
        if test_code:
            with open(test_path, "w") as f:
                f.write(test_code)
        elif os.path.exists(test_path):
            os.remove(test_path)

    def _run_command(self, args: list[str], timeout: int) -> SandboxResult:
        """Run a bal command and capture output."""
        try:
            result = subprocess.run(
                args,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            errors = self._parse_errors(output)
            return SandboxResult(
                success=(result.returncode == 0),
                errors=errors,
                output=output,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                errors=["Timeout expired"],
                output="Command timed out",
                return_code=-1,
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                errors=[str(e)],
                output=str(e),
                return_code=-1,
            )

    @staticmethod
    def _parse_errors(output: str) -> list[str]:
        """Extract ERROR lines from Ballerina compiler output."""
        errors = []
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("ERROR"):
                errors.append(stripped)
        return errors

    def compile_check(self, code: str) -> SandboxResult:
        """
        Compile-only check. Writes code to main.bal and runs `bal build`.
        No tests are written — this is a fast syntax + type check.
        """
        self._write_code(code, test_code=None)
        # Remove tests dir to avoid linking errors from stale test files
        test_dir = os.path.join(self.base_dir, "tests")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        result = self._run_command(
            [config.BAL_COMMAND, "build"],
            timeout=config.COMPILE_TIMEOUT,
        )
        # Re-create tests dir for future test runs
        os.makedirs(test_dir, exist_ok=True)
        logger.debug(
            "compile_check [%s]: success=%s errors=%s",
            self.sandbox_id, result.success, result.errors,
        )
        return result

    def test_check(self, code: str, test_code: str) -> SandboxResult:
        """
        Full test run. Writes code + tests and runs `bal test`.
        """
        self._write_code(code, test_code=test_code)
        result = self._run_command(
            [config.BAL_COMMAND, "test"],
            timeout=config.TEST_TIMEOUT,
        )
        logger.debug(
            "test_check [%s]: success=%s errors=%s",
            self.sandbox_id, result.success, result.errors,
        )
        return result

    def cleanup(self):
        """Remove the sandbox directory."""
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)

    def reset(self):
        """Reset the sandbox to a clean state."""
        self._ensure_project()
