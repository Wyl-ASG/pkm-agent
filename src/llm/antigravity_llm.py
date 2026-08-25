"""LLM wrapper driving local antigravitycli (agy) executable for text and structured JSON generation."""

import asyncio
import json
import logging
import os
import re
import shutil
from typing import Any, TypeVar
from pydantic import BaseModel, ValidationError

from src.config import settings
from src.llm.base import LLMProvider
from src.utils.resources import resource_manager

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def find_agy_executable(custom_path: str | None = None) -> str:
    """Find absolute path to agy CLI executable.

    Args:
        custom_path: Optional explicit binary path or name.

    Returns:
        String path to executable binary.

    Raises:
        FileNotFoundError: If the agy executable cannot be located.
    """
    if custom_path:
        resolved = shutil.which(custom_path)
        if resolved:
            return resolved
        if os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
            return custom_path
        raise FileNotFoundError(
            f"Specified agy binary path not found or not executable: {custom_path}"
        )

    agy_in_path = shutil.which("agy")
    if agy_in_path:
        return agy_in_path

    fallback_paths = [
        os.path.expanduser("~/.local/bin/agy"),
        "/usr/local/bin/agy",
        "/opt/homebrew/bin/agy",
    ]
    for path in fallback_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "The 'agy' (Antigravity CLI) executable was not found in PATH or standard directories (~/.local/bin/agy, /usr/local/bin/agy, /opt/homebrew/bin/agy).\n"
        "Note: If you are running inside a Docker container, 'agy' is a host binary and cannot be executed inside Linux Docker.\n"
        "To resolve this, run Qdrant in Docker and run the PKM app on your host machine."
    )


class AntigravityLLM(LLMProvider):
    """Wrapper for local antigravitycli (agy) executable implementing LLMProvider with process management."""

    def __init__(
        self,
        binary_path: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        timeout: int | None = None,
    ) -> None:
        """Initialize AntigravityLLM client.

        Args:
            binary_path: Optional path to agy binary. Defaults to auto-discovery.
            model: Optional model identifier (e.g., 'flash', 'pro').
            effort: Optional reasoning effort ('low', 'medium', 'high').
            timeout: Subprocess timeout in seconds. Defaults to settings.ANTIGRAVITY_TIMEOUT_SECONDS.
        """
        self.binary_path = find_agy_executable(binary_path or getattr(settings, "AGY_PATH", None))
        self.model = model or getattr(settings, "LLM_MODEL", None)
        self.effort = effort or getattr(settings, "LLM_EFFORT", None)
        self.timeout = timeout or getattr(settings, "ANTIGRAVITY_TIMEOUT_SECONDS", 120)

    def _clean_markdown_code_block(self, text: str) -> str:
        """Strip markdown triple backtick fences if present in JSON text."""
        cleaned = text.strip()
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2 and lines[-1].startswith("```"):
                return "\n".join(lines[1:-1]).strip()
            if lines[0].startswith("```"):
                return "\n".join(lines[1:]).strip()
        return cleaned

    def _extract_json_substring(self, text: str) -> str:
        """Extract outermost JSON object or array string from text if present."""
        cleaned = self._clean_markdown_code_block(text)
        if not cleaned:
            return ""

        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            pass

        start_obj = cleaned.find("{")
        end_obj = cleaned.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            candidate = cleaned[start_obj : end_obj + 1].strip()
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        start_arr = cleaned.find("[")
        end_arr = cleaned.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            candidate = cleaned[start_arr : end_arr + 1].strip()
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        return cleaned

    def _extract_json_payload(self, output_text: str) -> dict[str, Any] | str:
        """Extract structured JSON payload or string response from agy CLI envelope."""
        output_text = output_text.strip()
        if not output_text:
            return ""

        try:
            data = json.loads(output_text)
            if isinstance(data, dict):
                # 1. Check for structured_output
                if "structured_output" in data and data["structured_output"] is not None:
                    so = data["structured_output"]
                    if isinstance(so, dict):
                        return so
                    if isinstance(so, str) and so.strip():
                        extracted = self._extract_json_substring(so)
                        if extracted:
                            return extracted

                # 2. Check for response string
                if "response" in data and isinstance(data["response"], str) and data["response"].strip():
                    resp = self._extract_json_substring(data["response"])
                    if resp:
                        return resp

                # 3. Check if data is an agy CLI envelope
                is_agy_envelope = "status" in data or "conversation_id" in data
                if is_agy_envelope:
                    return ""

                # 4. If data is a dict payload directly (not agy envelope), return it
                return data
        except json.JSONDecodeError:
            pass

        return self._extract_json_substring(output_text)

    def _extract_error_message(self, stdout_bytes: bytes, stderr_bytes: bytes) -> str:
        """Extract human-readable error details from stdout and stderr of agy process.

        agy CLI outputs structured JSON errors (with an 'error' key) to stdout upon non-zero exit.
        """
        stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()
        stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()

        if stdout_str:
            try:
                data = json.loads(stdout_str)
                if isinstance(data, dict) and data.get("error"):
                    err = str(data["error"]).strip()
                    if stderr_str:
                        return f"{err} (stderr: {stderr_str[:500]})".strip()
                    return err
            except json.JSONDecodeError:
                pass

        if stderr_str and stdout_str:
            return f"{stderr_str[:1000]} | stdout: {stdout_str[:1000]}".strip()
        elif stderr_str:
            return stderr_str[:1000].strip()
        elif stdout_str:
            return stdout_str[:1000].strip()

        return "No error message provided on stdout or stderr"

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate plain unstructured text response using agy CLI.

        Args:
            prompt: Main user prompt or instruction.
            system_prompt: Optional system prompt to prepend.

        Returns:
            Generated text string response.
        """
        combined_prompt = prompt
        if system_prompt:
            combined_prompt = f"System Instructions:\n{system_prompt}\n\nUser Input Data:\n{prompt}"

        cmd = [
            self.binary_path,
            "--print",
            "-",
            "--output-format",
            "json",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.effort:
            cmd.extend(["--effort", self.effort])

        logger.info("Executing agy CLI command for text generation")
        async with resource_manager.antigravity_semaphore:
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=combined_prompt.encode("utf-8")),
                    timeout=self.timeout,
                )

                if process.returncode != 0:
                    err_msg = self._extract_error_message(stdout, stderr).strip()
                    logger.error("agy CLI command failed with exit code %d: %s", process.returncode, err_msg)
                    raise RuntimeError(f"agy CLI execution failed (exit code {process.returncode}): {err_msg}")

                output_str = stdout.decode("utf-8", errors="replace")
                try:
                    data = json.loads(output_str)
                    if isinstance(data, dict) and "response" in data:
                        return str(data["response"]).strip()
                except json.JSONDecodeError as err:
                    logger.warning("LLM generated invalid JSON format. Falling back to raw string. Error: %s", err)

                return output_str.strip()

            except asyncio.TimeoutError:
                if process:
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        pass
                logger.error("agy CLI command execution timed out after %d seconds", self.timeout)
                raise TimeoutError(f"agy CLI execution timed out after {self.timeout} seconds")
            except Exception as err:
                if process and process.returncode is None:
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        pass
                logger.exception("Unexpected error during agy CLI text generation: %s", err)
                raise

    async def _exec_agy_subprocess(
        self,
        cmd: list[str],
        combined_prompt: str,
    ) -> tuple[int, bytes, bytes]:
        """Execute agy subprocess wrapped with resource semaphore and cleanup handling."""
        async with resource_manager.antigravity_semaphore:
            process = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=combined_prompt.encode("utf-8")),
                    timeout=self.timeout,
                )
                return process.returncode, stdout, stderr
            except Exception:
                if process and process.returncode is None:
                    try:
                        process.kill()
                        await process.wait()
                    except Exception:
                        pass
                raise

    async def generate_json(
        self,
        prompt: str,
        schema_model: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generate structured output adhering to a Pydantic v2 schema using agy CLI.

        Args:
            prompt: User request or text content to process.
            schema_model: Pydantic v2 BaseModel class defining expected output schema.
            system_prompt: Optional instructions for system behavior.

        Returns:
            Validated Pydantic v2 model instance of type T.
        """
        schema_dict = schema_model.model_json_schema()
        schema_json = json.dumps(schema_dict)

        combined_prompt = prompt
        if system_prompt:
            combined_prompt = f"System Instructions:\n{system_prompt}\n\nUser Input Data:\n{prompt}"

        cmd = [
            self.binary_path,
            "--print",
            "-",
            "--json-schema",
            schema_json,
            "--output-format",
            "json",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.effort:
            cmd.extend(["--effort", self.effort])

        logger.info("Executing agy CLI command for structured JSON generation")
        last_err: Exception | None = None
        try:
            returncode, stdout, stderr = await self._exec_agy_subprocess(cmd, combined_prompt)
            if returncode == 0:
                output_str = stdout.decode("utf-8", errors="replace")
                payload = self._extract_json_payload(output_str)
                if isinstance(payload, dict):
                    return schema_model.model_validate(payload)
                elif isinstance(payload, str) and payload.strip():
                    return schema_model.model_validate_json(payload)
            else:
                err_msg = self._extract_error_message(stdout, stderr).strip()
                logger.warning("agy CLI command with --json-schema exited with code %d: %s", returncode, err_msg)
        except asyncio.TimeoutError:
            logger.error("agy CLI execution timed out after %d seconds", self.timeout)
            raise TimeoutError(f"agy CLI execution timed out after {self.timeout} seconds")
        except (json.JSONDecodeError, ValueError, ValidationError) as parse_err:
            logger.warning("Initial agy CLI structured output validation failed for %s: %s", schema_model.__name__, parse_err)
            last_err = parse_err
        except Exception as err:
            logger.warning("Initial agy CLI execution failed for %s: %s", schema_model.__name__, err)
            last_err = err

        # Fallback attempt: Request JSON explicitly in prompt without --json-schema
        logger.info("Retrying agy CLI generation with explicit JSON prompt fallback")
        fallback_prompt = (
            f"{combined_prompt}\n\n"
            "CRITICAL REQUIREMENT: Respond ONLY with a valid JSON object matching this schema. "
            "Do NOT include markdown backticks or any conversation:\n"
            f"{schema_json}"
        )
        fallback_cmd = [
            self.binary_path,
            "--print",
            "-",
            "--output-format",
            "json",
        ]
        if self.model:
            fallback_cmd.extend(["--model", self.model])
        if self.effort:
            fallback_cmd.extend(["--effort", self.effort])

        try:
            returncode, stdout, stderr = await self._exec_agy_subprocess(fallback_cmd, fallback_prompt)
            if returncode != 0:
                err_msg = self._extract_error_message(stdout, stderr).strip()
                raise RuntimeError(f"agy CLI execution failed (exit code {returncode}): {err_msg}")

            output_str = stdout.decode("utf-8", errors="replace")
            payload = self._extract_json_payload(output_str)
            if isinstance(payload, dict):
                return schema_model.model_validate(payload)
            elif isinstance(payload, str) and payload.strip():
                return schema_model.model_validate_json(payload)
            else:
                raise ValueError(f"Extracted payload was empty or invalid string: {payload!r}")
        except asyncio.TimeoutError:
            logger.error("Fallback agy CLI execution timed out after %d seconds", self.timeout)
            raise TimeoutError(f"Fallback agy CLI execution timed out after {self.timeout} seconds")
        except Exception as err:
            logger.exception("Failed to parse JSON response from agy CLI output after fallback: %s", err)
            if last_err:
                raise ValueError(f"Failed to generate JSON for model {schema_model.__name__}: {err}") from last_err
            raise


__all__ = ["AntigravityLLM", "find_agy_executable"]
