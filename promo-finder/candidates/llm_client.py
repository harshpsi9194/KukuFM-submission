"""
llm_client.py — Model-Agnostic LLM Client for Promo Finder Pipeline.

Provides a unified, provider-agnostic interface for executing prompts:
1. REST API Mode:
   - When OPENAI_API_KEY / GEMINI_API_KEY / LLM_API_KEY is configured in the environment,
     makes direct standard HTTPS REST API calls (OpenAI-compatible or Gemini format).
2. Local CLI / Sandbox Bridge Mode:
   - When running in local sandbox without external API keys, routes through AGY_EXE / PROMO_CLI.
3. Model Parameterization:
   - Model identifier is passed per call or defaulted via PROMO_MODEL environment variable.
   - Swapping models is a 1-line configuration change (e.g. model="gpt-4o", model="gemini-3.8-flash-low").
4. Telemetry & Cost Accounting:
   - Every call logs tokens, latency, cost ($0.075/1M in, $0.30/1M out), stage, candidate IDs, and beat IDs
     to outputs/llm_calls.jsonl (successes) or outputs/llm_call_errors.jsonl (failures).
5. Robustness:
   - Exponential backoff on rate limits / 429 errors.
   - Markdown JSON fence stripping and robust JSON payload extraction.
"""

import datetime
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent

import shutil

def get_default_model() -> str:
    """Dynamically determine default model based on available environment credentials."""
    if os.environ.get("PROMO_MODEL"):
        return os.environ["PROMO_MODEL"]
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini-2.5-flash"
    return "gemini-2.5-flash"

DEFAULT_MODEL = get_default_model()
AGY_EXE = os.environ.get("AGY_EXE", shutil.which("agy") or str(Path.home() / ".gemini" / "bin" / "agy"))
API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", os.environ.get("GEMINI_API_KEY", "")))
API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "")

def get_api_credentials() -> Tuple[str, str]:
    """Dynamically resolve API credentials and base URL at call time."""
    api_key = (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or API_KEY
    )
    base_url = os.environ.get("LLM_API_BASE_URL") or API_BASE_URL
    if not base_url and api_key:
        if os.environ.get("OPENAI_API_KEY") or api_key.startswith("sk-"):
            base_url = "https://api.openai.com/v1"
        elif os.environ.get("GEMINI_API_KEY") or api_key.startswith("AIza"):
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        else:
            base_url = "https://api.openai.com/v1"
    return api_key, base_url

# Pricing assumptions (Gemini Flash class: $0.075/1M in, $0.30/1M out)
INPUT_COST_PER_MILLION = 0.075
OUTPUT_COST_PER_MILLION = 0.30

# Logging paths
SUCCESS_LOG_PATH = ROOT_DIR / "outputs" / "llm_calls.jsonl"
ERROR_LOG_PATH = ROOT_DIR / "outputs" / "llm_call_errors.jsonl"


def set_log_dir(log_dir: Path):
    """Directs telemetry logs to a dedicated directory (e.g. outputs_final_batch)."""
    global SUCCESS_LOG_PATH, ERROR_LOG_PATH
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    SUCCESS_LOG_PATH = log_dir / "llm_calls.jsonl"
    ERROR_LOG_PATH = log_dir / "llm_call_errors.jsonl"

log_lock = threading.Lock()


def estimate_tokens(text: str) -> int:
    """Estimates token count from text using 1.33 tokens/word heuristic."""
    words = len(text.split())
    chars = len(text)
    return max(1, int(words * 1.33)) if words > 0 else max(1, chars // 4)


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculates estimated USD cost from token counts."""
    in_cost = (input_tokens / 1_000_000.0) * INPUT_COST_PER_MILLION
    out_cost = (output_tokens / 1_000_000.0) * OUTPUT_COST_PER_MILLION
    return round(in_cost + out_cost, 7)


def extract_json_payload(raw_text: str) -> Any:
    """
    Extracts and parses JSON object or array from LLM response text,
    stripping markdown fences or surrounding commentary.
    """
    raw = raw_text.strip()
    # Match markdown code block with array or object
    m = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Match raw JSON array or object
    m = re.search(r"([\[\{].*[\}\]])", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return json.loads(raw)


def log_llm_call(log_path: Path, entry: Dict[str, Any]):
    """Thread-safe append of JSON line to audit log."""
    with log_lock:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def _call_http_api(prompt: str, model: str, api_url: str, api_key: str, timeout: int = 120) -> str:
    """Standard OpenAI-compatible HTTPS REST API call using standard library urllib."""
    url = f"{api_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert microdrama promo analysis model. Always output strict JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        resp_data = json.loads(response.read().decode("utf-8"))
        return resp_data["choices"][0]["message"]["content"]


def _call_cli_bridge(prompt: str, model: str, cli_exe: str, timeout: int = 120) -> str:
    """Local CLI bridge invocation (agy / local runner)."""
    resolved_exe = cli_exe if os.path.exists(cli_exe) else shutil.which(cli_exe)
    if not resolved_exe:
        raise RuntimeError(
            f"CLI runner executable '{cli_exe}' not found. "
            "Please configure OPENAI_API_KEY / GEMINI_API_KEY for REST API mode, "
            "or set AGY_EXE to the path of your local CLI bridge."
        )
    res = subprocess.run(
        [resolved_exe, "--model", model, "--print", prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined_err = (res.stdout + " " + res.stderr).lower()
    if any(k in combined_err for k in ["resource_exhausted", "quota", "rate limit", "429"]):
        raise RuntimeError(f"Rate limit / quota exceeded (HTTP 429): {combined_err[:120]}")
    if res.returncode != 0 and not res.stdout.strip():
        raise RuntimeError(f"CLI command failed with code {res.returncode}: {res.stderr}")
    return res.stdout.strip()


def call_llm(
    prompt: str,
    prompt_file: str,
    stage: str = "general",
    episode_number: int = 1,
    candidate_ids: Optional[List[str]] = None,
    beat_ids: Optional[List[str]] = None,
    model: Optional[str] = None,
    max_retries: int = 4,
    success_log: Optional[Path] = None,
    error_log: Optional[Path] = None,
) -> Any:
    """
    Executes a prompt through the model-agnostic client, parses JSON, and logs telemetry.
    Model is fully parameterized and easily configured.
    """
    active_model = model or get_default_model()
    active_api_key, active_base_url = get_api_credentials()
    s_log = success_log or SUCCESS_LOG_PATH
    e_log = error_log or ERROR_LOG_PATH

    c_ids = candidate_ids or []
    b_ids = beat_ids or []

    input_tokens = estimate_tokens(prompt)
    t0 = time.time()
    last_err = None

    for attempt in range(max_retries):
        try:
            # Mode 1: Standard REST API if active_base_url and active_api_key are set
            if active_base_url and active_api_key:
                raw_output = _call_http_api(prompt, active_model, active_base_url, active_api_key)
            # Mode 2: Local CLI bridge runner
            else:
                raw_output = _call_cli_bridge(prompt, active_model, AGY_EXE)

            parsed = extract_json_payload(raw_output)
            latency = round(time.time() - t0, 3)
            output_tokens = estimate_tokens(raw_output)
            cost = calculate_cost(input_tokens, output_tokens)

            log_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "episode_number": episode_number,
                "stage": stage,
                "model": active_model,
                "prompt_file": prompt_file,
                "candidate_ids": c_ids,
                "beat_ids": b_ids,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_seconds": latency,
                "cost_usd": cost,
            }
            log_llm_call(s_log, log_entry)
            return parsed

        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "quota" in err_str or "rate limit" in err_str
            if is_rate_limit and attempt < max_retries - 1:
                backoff = (2 ** attempt) * 4
                print(f"[RATE LIMIT] Backing off {backoff}s on {prompt_file} (attempt {attempt+1}/{max_retries})...")
                time.sleep(backoff)
                continue
            elif attempt < max_retries - 1:
                backoff = (2 ** attempt) * 2
                time.sleep(backoff)
            else:
                latency = round(time.time() - t0, 3)
                err_entry = {
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "episode_number": episode_number,
                    "stage": stage,
                    "model": active_model,
                    "prompt_file": prompt_file,
                    "candidate_ids": c_ids,
                    "beat_ids": b_ids,
                    "input_tokens": input_tokens,
                    "latency_seconds": latency,
                    "error": str(e),
                }
                log_llm_call(e_log, err_entry)
                raise e

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")
