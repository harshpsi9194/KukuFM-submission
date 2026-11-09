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
    # Local CLI Bridge default
    return "gemini-3.8-flash-low"

DEFAULT_MODEL = get_default_model()
AGY_EXE = os.environ.get("AGY_EXE", shutil.which("agy") or str(Path.home() / ".gemini" / "bin" / "agy"))
API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", os.environ.get("GEMINI_API_KEY", "")))
API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "")


# -----------------------------------------------------------------------------
# Comprehensive Multi-Provider Model Catalog & Pricing Registry
# -----------------------------------------------------------------------------
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    # Google Gemini Models
    "gemini-2.5-flash": {
        "provider": "google",
        "description": "Google Gemini 2.5 Flash (Flagship high-speed multimodal)",
        "input_cost_per_million": 0.075,
        "output_cost_per_million": 0.30,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "gemini-2.5-flash",
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "description": "Google Gemini 2.5 Pro (Deep reasoning and complex synthesis)",
        "input_cost_per_million": 1.25,
        "output_cost_per_million": 5.00,
        "cli_equivalent": "gemini-3.8-flash-high",
        "rest_equivalent": "gemini-2.5-pro",
    },
    "gemini-2.0-flash": {
        "provider": "google",
        "description": "Google Gemini 2.0 Flash (Next-gen ultra-low latency)",
        "input_cost_per_million": 0.10,
        "output_cost_per_million": 0.40,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "gemini-2.0-flash",
    },
    "gemini-2.0-flash-lite": {
        "provider": "google",
        "description": "Google Gemini 2.0 Flash Lite (Cost-efficient lightweight)",
        "input_cost_per_million": 0.075,
        "output_cost_per_million": 0.30,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "gemini-2.0-flash-lite",
    },
    "gemini-2.0-flash-thinking-exp": {
        "provider": "google",
        "description": "Google Gemini 2.0 Flash Thinking Experimental",
        "input_cost_per_million": 0.10,
        "output_cost_per_million": 0.40,
        "cli_equivalent": "gemini-3.8-flash-medium",
        "rest_equivalent": "gemini-2.0-flash-thinking-exp",
    },
    "gemini-1.5-flash": {
        "provider": "google",
        "description": "Google Gemini 1.5 Flash (Standard production workhorse)",
        "input_cost_per_million": 0.075,
        "output_cost_per_million": 0.30,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "gemini-1.5-flash",
    },
    "gemini-1.5-flash-8b": {
        "provider": "google",
        "description": "Google Gemini 1.5 Flash 8B (Sub-10B lightweight tier)",
        "input_cost_per_million": 0.0375,
        "output_cost_per_million": 0.15,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "gemini-1.5-flash-8b",
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "description": "Google Gemini 1.5 Pro (2M context reasoning)",
        "input_cost_per_million": 1.25,
        "output_cost_per_million": 5.00,
        "cli_equivalent": "gemini-3.8-flash-high",
        "rest_equivalent": "gemini-1.5-pro",
    },
    "gemini-3.8-flash-low": {
        "provider": "google",
        "description": "Local CLI Gemini 3.8 Flash (Low Reasoning Effort)",
        "input_cost_per_million": 0.075,
        "output_cost_per_million": 0.30,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "gemini-2.5-flash",
    },
    "gemini-3.8-flash-medium": {
        "provider": "google",
        "description": "Local CLI Gemini 3.8 Flash (Medium Reasoning Effort)",
        "input_cost_per_million": 0.075,
        "output_cost_per_million": 0.30,
        "cli_equivalent": "gemini-3.8-flash-medium",
        "rest_equivalent": "gemini-2.5-flash",
    },
    "gemini-3.8-flash-high": {
        "provider": "google",
        "description": "Local CLI Gemini 3.8 Flash (High Reasoning Effort)",
        "input_cost_per_million": 0.075,
        "output_cost_per_million": 0.30,
        "cli_equivalent": "gemini-3.8-flash-high",
        "rest_equivalent": "gemini-2.5-pro",
    },

    # OpenAI Models
    "gpt-4o": {
        "provider": "openai",
        "description": "OpenAI GPT-4o (Flagship Omni model)",
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 10.00,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "gpt-4o",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "description": "OpenAI GPT-4o Mini (Cost-efficient multimodal)",
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "gpt-4o-mini",
    },
    "gpt-4o-2024-11-20": {
        "provider": "openai",
        "description": "OpenAI GPT-4o Checkpoint (2024-11-20)",
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 10.00,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "gpt-4o-2024-11-20",
    },
    "gpt-4o-2024-08-06": {
        "provider": "openai",
        "description": "OpenAI GPT-4o Structured Outputs Checkpoint",
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 10.00,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "gpt-4o-2024-08-06",
    },
    "chatgpt-4o-latest": {
        "provider": "openai",
        "description": "OpenAI ChatGPT-4o Latest Dynamic Endpoint",
        "input_cost_per_million": 5.00,
        "output_cost_per_million": 15.00,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "chatgpt-4o-latest",
    },
    "gpt-4-turbo": {
        "provider": "openai",
        "description": "OpenAI GPT-4 Turbo (128k context)",
        "input_cost_per_million": 10.00,
        "output_cost_per_million": 30.00,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "gpt-4-turbo",
    },
    "gpt-4": {
        "provider": "openai",
        "description": "OpenAI GPT-4 Classic",
        "input_cost_per_million": 30.00,
        "output_cost_per_million": 60.00,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "gpt-4",
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "description": "OpenAI GPT-3.5 Turbo",
        "input_cost_per_million": 0.50,
        "output_cost_per_million": 1.50,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "gpt-3.5-turbo",
    },
    "o1": {
        "provider": "openai",
        "description": "OpenAI o1 Reasoning Model",
        "input_cost_per_million": 15.00,
        "output_cost_per_million": 60.00,
        "cli_equivalent": "claude-opus-4-6-thinking",
        "rest_equivalent": "o1",
    },
    "o1-mini": {
        "provider": "openai",
        "description": "OpenAI o1 Mini Fast Reasoning Model",
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 12.00,
        "cli_equivalent": "claude-sonnet-4-6",
        "rest_equivalent": "o1-mini",
    },
    "o1-preview": {
        "provider": "openai",
        "description": "OpenAI o1 Preview Reasoning Model",
        "input_cost_per_million": 15.00,
        "output_cost_per_million": 60.00,
        "cli_equivalent": "claude-opus-4-6-thinking",
        "rest_equivalent": "o1-preview",
    },
    "o3-mini": {
        "provider": "openai",
        "description": "OpenAI o3 Mini Efficient Reasoning Model",
        "input_cost_per_million": 1.10,
        "output_cost_per_million": 4.40,
        "cli_equivalent": "claude-sonnet-4-6",
        "rest_equivalent": "o3-mini",
    },
    "gpt-oss-120b-medium": {
        "provider": "openai/oss",
        "description": "Local CLI Open-Source 120B Foundation Model",
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "gpt-4o-mini",
    },

    # Anthropic Claude Models
    "claude-3-7-sonnet": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3.7 Sonnet (Hybrid Reasoning)",
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "cli_equivalent": "claude-sonnet-4-6",
        "rest_equivalent": "claude-3-7-sonnet",
    },
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3.5 Sonnet (State of the art intelligence)",
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "cli_equivalent": "claude-sonnet-4-6",
        "rest_equivalent": "claude-3-5-sonnet-20241022",
    },
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3.5 Sonnet (20241022 release)",
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "cli_equivalent": "claude-sonnet-4-6",
        "rest_equivalent": "claude-3-5-sonnet-20241022",
    },
    "claude-3-5-sonnet-20240620": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3.5 Sonnet (Initial release)",
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "cli_equivalent": "claude-sonnet-4-6",
        "rest_equivalent": "claude-3-5-sonnet-20240620",
    },
    "claude-3-5-haiku": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3.5 Haiku (High speed & low cost)",
        "input_cost_per_million": 0.80,
        "output_cost_per_million": 4.00,
        "cli_equivalent": "claude-sonnet-4-6",
        "rest_equivalent": "claude-3-5-haiku-20241022",
    },
    "claude-3-5-haiku-20241022": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3.5 Haiku (20241022 release)",
        "input_cost_per_million": 0.80,
        "output_cost_per_million": 4.00,
        "cli_equivalent": "claude-sonnet-4-6",
        "rest_equivalent": "claude-3-5-haiku-20241022",
    },
    "claude-3-opus": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3 Opus (Complex reasoning and creativity)",
        "input_cost_per_million": 15.00,
        "output_cost_per_million": 75.00,
        "cli_equivalent": "claude-opus-4-6-thinking",
        "rest_equivalent": "claude-3-opus-20240229",
    },
    "claude-3-opus-20240229": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3 Opus (20240229 release)",
        "input_cost_per_million": 15.00,
        "output_cost_per_million": 75.00,
        "cli_equivalent": "claude-opus-4-6-thinking",
        "rest_equivalent": "claude-3-opus-20240229",
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "description": "Anthropic Claude 3 Haiku",
        "input_cost_per_million": 0.25,
        "output_cost_per_million": 1.25,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "claude-3-haiku-20240307",
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "description": "Local CLI Claude Sonnet 4.6 (Thinking)",
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
        "cli_equivalent": "claude-sonnet-4-6",
        "rest_equivalent": "claude-3-5-sonnet-20241022",
    },
    "claude-opus-4-6-thinking": {
        "provider": "anthropic",
        "description": "Local CLI Claude Opus 4.6 (Extended Thinking)",
        "input_cost_per_million": 15.00,
        "output_cost_per_million": 75.00,
        "cli_equivalent": "claude-opus-4-6-thinking",
        "rest_equivalent": "claude-3-opus-20240229",
    },

    # DeepSeek Models
    "deepseek-chat": {
        "provider": "deepseek",
        "description": "DeepSeek-V3 671B MoE General Model",
        "input_cost_per_million": 0.14,
        "output_cost_per_million": 0.28,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "deepseek-chat",
    },
    "deepseek-reasoner": {
        "provider": "deepseek",
        "description": "DeepSeek-R1 Open-Weights Reasoning Model",
        "input_cost_per_million": 0.55,
        "output_cost_per_million": 2.19,
        "cli_equivalent": "claude-opus-4-6-thinking",
        "rest_equivalent": "deepseek-reasoner",
    },
    "deepseek-v3": {
        "provider": "deepseek",
        "description": "DeepSeek-V3 MoE Checkpoint",
        "input_cost_per_million": 0.14,
        "output_cost_per_million": 0.28,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "deepseek-v3",
    },
    "deepseek-r1": {
        "provider": "deepseek",
        "description": "DeepSeek-R1 Checkpoint",
        "input_cost_per_million": 0.55,
        "output_cost_per_million": 2.19,
        "cli_equivalent": "claude-opus-4-6-thinking",
        "rest_equivalent": "deepseek-r1",
    },

    # Meta Llama Models
    "meta-llama/Llama-3.3-70B-Instruct": {
        "provider": "meta",
        "description": "Meta Llama 3.3 70B Instruct (High efficiency flagship)",
        "input_cost_per_million": 0.13,
        "output_cost_per_million": 0.40,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "meta-llama/Llama-3.1-405B-Instruct": {
        "provider": "meta",
        "description": "Meta Llama 3.1 405B Instruct (Ultra-scale open model)",
        "input_cost_per_million": 1.00,
        "output_cost_per_million": 3.00,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "meta-llama/Llama-3.1-405B-Instruct",
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "provider": "meta",
        "description": "Meta Llama 3.1 70B Instruct",
        "input_cost_per_million": 0.13,
        "output_cost_per_million": 0.40,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "meta-llama/Llama-3.1-70B-Instruct",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "provider": "meta",
        "description": "Meta Llama 3.1 8B Instruct",
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.10,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "llama-3.3-70b": {
        "provider": "meta",
        "description": "Llama 3.3 70B Short Alias",
        "input_cost_per_million": 0.13,
        "output_cost_per_million": 0.40,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "llama-3.1-70b": {
        "provider": "meta",
        "description": "Llama 3.1 70B Short Alias",
        "input_cost_per_million": 0.13,
        "output_cost_per_million": 0.40,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "meta-llama/Llama-3.1-70B-Instruct",
    },
    "llama-3.1-8b": {
        "provider": "meta",
        "description": "Llama 3.1 8B Short Alias",
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.10,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "meta-llama/Llama-3.1-8B-Instruct",
    },

    # Mistral AI Models
    "mistral-large-latest": {
        "provider": "mistral",
        "description": "Mistral Large 2 (Top tier reasoning & multilingual)",
        "input_cost_per_million": 2.00,
        "output_cost_per_million": 6.00,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "mistral-large-latest",
    },
    "mistral-small-latest": {
        "provider": "mistral",
        "description": "Mistral Small (Cost-effective fast reasoning)",
        "input_cost_per_million": 0.20,
        "output_cost_per_million": 0.60,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "mistral-small-latest",
    },
    "codestral-latest": {
        "provider": "mistral",
        "description": "Mistral Codestral (Code & JSON structured output specialist)",
        "input_cost_per_million": 0.30,
        "output_cost_per_million": 0.90,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "codestral-latest",
    },
    "open-mixtral-8x22b": {
        "provider": "mistral",
        "description": "Mistral Mixtral 8x22B MoE",
        "input_cost_per_million": 0.90,
        "output_cost_per_million": 0.90,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "open-mixtral-8x22b",
    },
    "open-mixtral-8x7b": {
        "provider": "mistral",
        "description": "Mistral Mixtral 8x7B MoE",
        "input_cost_per_million": 0.45,
        "output_cost_per_million": 0.45,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "open-mixtral-8x7b",
    },

    # Qwen (Alibaba) Models
    "qwen-2.5-72b-instruct": {
        "provider": "qwen",
        "description": "Qwen 2.5 72B Instruct (Leading open model)",
        "input_cost_per_million": 0.35,
        "output_cost_per_million": 0.70,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "Qwen/Qwen2.5-72B-Instruct",
    },
    "qwen-2.5-32b-instruct": {
        "provider": "qwen",
        "description": "Qwen 2.5 32B Instruct (High efficiency)",
        "input_cost_per_million": 0.20,
        "output_cost_per_million": 0.40,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "Qwen/Qwen2.5-32B-Instruct",
    },
    "qwen-2.5-coder-32b-instruct": {
        "provider": "qwen",
        "description": "Qwen 2.5 Coder 32B (JSON & code generation specialist)",
        "input_cost_per_million": 0.20,
        "output_cost_per_million": 0.40,
        "cli_equivalent": "gpt-oss-120b-medium",
        "rest_equivalent": "Qwen/Qwen2.5-Coder-32B-Instruct",
    },
    "qwen-2.5-7b-instruct": {
        "provider": "qwen",
        "description": "Qwen 2.5 7B Instruct (Compact edge model)",
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.10,
        "cli_equivalent": "gemini-3.8-flash-low",
        "rest_equivalent": "Qwen/Qwen2.5-7b-instruct",
    },
}


_DETECTED_CLI_MODELS: Optional[set] = None


def get_available_cli_models(cli_exe: str = AGY_EXE) -> set:
    """Discovers available models from local CLI bridge (agy) dynamically."""
    global _DETECTED_CLI_MODELS
    if _DETECTED_CLI_MODELS is not None:
        return _DETECTED_CLI_MODELS
    resolved = cli_exe if os.path.exists(cli_exe) else shutil.which(cli_exe)
    if resolved:
        try:
            res = subprocess.run([resolved, "models"], capture_output=True, text=True, timeout=5)
            found = set()
            for line in res.stdout.splitlines():
                parts = line.strip().split()
                if parts and not parts[0].startswith("Fetching") and not parts[0].startswith("⠋"):
                    found.add(parts[0].lower())
            if found:
                _DETECTED_CLI_MODELS = found
                return _DETECTED_CLI_MODELS
        except Exception:
            pass
    # Fallback to standard known set
    _DETECTED_CLI_MODELS = {
        "gemini-3.8-flash-low",
        "gemini-3.8-flash-medium",
        "gemini-3.8-flash-high",
        "claude-sonnet-4-6",
        "claude-opus-4-6-thinking",
        "gpt-oss-120b-medium",
    }
    return _DETECTED_CLI_MODELS


def resolve_cli_model(model: str) -> str:
    """Maps generic / third-party model names to supported local CLI bridge (agy) models."""
    m = model.lower().strip()
    available = get_available_cli_models()
    if m in available:
        return m
    if model in MODEL_CATALOG and "cli_equivalent" in MODEL_CATALOG[model]:
        return MODEL_CATALOG[model]["cli_equivalent"]
    # Keyword / family fallbacks
    if any(k in m for k in ["opus", "reasoner", "r1", "o1", "thinking"]):
        return "claude-opus-4-6-thinking" if "claude-opus-4-6-thinking" in available else "gemini-3.8-flash-high"
    if any(k in m for k in ["claude", "sonnet"]):
        return "claude-sonnet-4-6" if "claude-sonnet-4-6" in available else "gemini-3.8-flash-medium"
    if any(k in m for k in ["gpt", "llama", "deepseek", "qwen", "mistral", "oss"]):
        return "gpt-oss-120b-medium" if "gpt-oss-120b-medium" in available else "gemini-3.8-flash-low"
    if any(k in m for k in ["pro", "high"]):
        return "gemini-3.8-flash-high"
    if "medium" in m:
        return "gemini-3.8-flash-medium"
    return "gemini-3.8-flash-low"


def resolve_rest_model(model: str, base_url: str) -> str:
    """Maps local/generic model names to supported public REST API model identifiers."""
    m = model.strip()
    if model in MODEL_CATALOG and "rest_equivalent" in MODEL_CATALOG[model]:
        m = MODEL_CATALOG[model]["rest_equivalent"]
    if "generativelanguage.googleapis.com" in base_url:
        if "gemini-3.8" in m:
            return "gemini-2.5-flash"
    elif "api.openai.com" in base_url:
        if "gemini" in m.lower():
            return "gpt-4o-mini"
    return m

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


def calculate_cost(input_tokens: int, output_tokens: int, model: Optional[str] = None) -> float:
    """Calculates estimated USD cost from token counts and model pricing."""
    in_rate = INPUT_COST_PER_MILLION
    out_rate = OUTPUT_COST_PER_MILLION
    if model and model in MODEL_CATALOG:
        in_rate = MODEL_CATALOG[model].get("input_cost_per_million", in_rate)
        out_rate = MODEL_CATALOG[model].get("output_cost_per_million", out_rate)
    in_cost = (input_tokens / 1_000_000.0) * in_rate
    out_cost = (output_tokens / 1_000_000.0) * out_rate
    return round(in_cost + out_cost, 7)


def list_supported_models() -> List[Dict[str, Any]]:
    """Returns all models registered in the catalog with their provider, description, and pricing."""
    return [
        {
            "model": k,
            **v
        }
        for k, v in MODEL_CATALOG.items()
    ]


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
    active_model = resolve_rest_model(model, api_url)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": active_model,
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
    cli_model = resolve_cli_model(model)
    res = subprocess.run(
        [resolved_exe, "--model", cli_model, "--print", prompt],
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
            cost = calculate_cost(input_tokens, output_tokens, active_model)

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
