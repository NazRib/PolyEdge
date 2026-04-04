"""
LLM Provider Abstraction
Routes forecasting and news-search calls to either Claude (Anthropic) or
GPT-5.4 (Azure AI Foundry Responses API) so the rest of the codebase stays
provider-agnostic.

Two call shapes are supported:
    1. call_llm()           — structured forecasting (no web search)
    2. call_llm_with_search — news retrieval with web grounding

Environment variables:
    Claude:  ANTHROPIC_API_KEY
    GPT:     AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
"""

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ─── Provider constants ──────────────────────────────────

PROVIDER_CLAUDE = "claude"
PROVIDER_GPT = "gpt"
VALID_PROVIDERS = {PROVIDER_CLAUDE, PROVIDER_GPT}

# ─── Anthropic config ────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ─── Azure OpenAI config ─────────────────────────────────

# Set these via environment or config.py
DEFAULT_GPT_DEPLOYMENT = "gpt-5.4-pro"   # Azure deployment name


def _is_reasoning_model(model_name: str) -> bool:
    """
    Reasoning / 'pro' models don't support temperature, top_p, or
    similar sampling params.  Detect them by name pattern.
    """
    m = model_name.lower()
    if m.startswith(("o1", "o3")):
        return True
    if "-pro" in m:
        return True
    if "reasoning" in m:
        return True
    return False


def _get_azure_config() -> dict:
    """Gather Azure OpenAI settings from env vars."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", DEFAULT_GPT_DEPLOYMENT)
    return {"endpoint": endpoint, "api_key": api_key, "deployment": deployment}


def validate_provider(provider: str) -> str:
    """Normalise and validate provider string."""
    p = provider.lower().strip()
    if p in ("gpt", "openai", "azure", "gpt-5.4", "gpt-5.4-pro"):
        return PROVIDER_GPT
    if p in ("claude", "anthropic"):
        return PROVIDER_CLAUDE
    raise ValueError(
        f"Unknown LLM provider '{provider}'. Use 'claude' or 'gpt'."
    )


def provider_ready(provider: str) -> tuple[bool, str]:
    """Check whether the required credentials are set for *provider*."""
    p = validate_provider(provider)
    if p == PROVIDER_CLAUDE:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return True, "Claude API key set"
        return False, "Set ANTHROPIC_API_KEY to use Claude"
    # GPT
    cfg = _get_azure_config()
    missing = []
    if not cfg["api_key"]:
        missing.append("AZURE_OPENAI_API_KEY")
    if not cfg["endpoint"]:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if missing:
        return False, f"Set {', '.join(missing)} to use GPT"
    return True, f"Azure OpenAI ready (deployment: {cfg['deployment']})"


def model_tag_for_provider(provider: str) -> str:
    """Return a short, stable tag suitable for paper-trade comparison."""
    p = validate_provider(provider)
    if p == PROVIDER_CLAUDE:
        return DEFAULT_CLAUDE_MODEL.split("-202")[0]  # e.g. "claude-sonnet-4"
    cfg = _get_azure_config()
    return cfg["deployment"]  # e.g. "gpt-5.4"


# ═════════════════════════════════════════════════════════
#  UNIFIED DISPATCH
# ═════════════════════════════════════════════════════════

def call_llm(
    user_prompt: str,
    system_prompt: str = "",
    provider: str = PROVIDER_CLAUDE,
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Send a prompt to the configured LLM and return the text response.

    This is the **non-search** path used for probability forecasting.
    """
    p = validate_provider(provider)
    if p == PROVIDER_CLAUDE:
        return _call_claude(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model=model or DEFAULT_CLAUDE_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            max_retries=max_retries,
        )
    return _call_gpt(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        max_retries=max_retries,
        use_web_search=False,
    )


def call_llm_with_search(
    user_prompt: str,
    system_prompt: str = "",
    provider: str = PROVIDER_CLAUDE,
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.1,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """
    Send a prompt with web-search grounding enabled.

    Used by NewsEnricher to fetch recent information.
    """
    p = validate_provider(provider)
    if p == PROVIDER_CLAUDE:
        return _call_claude_with_search(
            user_prompt=user_prompt,
            model=model or DEFAULT_CLAUDE_MODEL,
            api_key=api_key,
            max_retries=max_retries,
        )
    return _call_gpt(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
        max_retries=max_retries,
        use_web_search=True,
    )


# ═════════════════════════════════════════════════════════
#  CLAUDE (Anthropic)
# ═════════════════════════════════════════════════════════

def _call_claude(
    user_prompt: str,
    system_prompt: str,
    model: str = DEFAULT_CLAUDE_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """Direct Claude Messages API call (no web search)."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("No ANTHROPIC_API_KEY set — returning None")
        return None

    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30,
            )
            if resp.status_code == 429:
                wait = max(
                    int(resp.headers.get("retry-after", 2 ** (attempt + 1))),
                    2 ** (attempt + 1),
                )
                logger.info(f"Claude rate-limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            text = "".join(
                b["text"] for b in data.get("content", []) if b.get("type") == "text"
            )
            return text.strip()

        except requests.exceptions.HTTPError:
            logger.error(f"Claude API error: {resp.status_code} {resp.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return None

    logger.warning("Claude: max retries exhausted")
    return None


def _call_claude_with_search(
    user_prompt: str,
    model: str = DEFAULT_CLAUDE_MODEL,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[str]:
    """Claude Messages API with web_search tool enabled."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("No ANTHROPIC_API_KEY set — skipping news search")
        return None

    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 1000,
        "temperature": 0.1,
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        "messages": [{"role": "user", "content": user_prompt}],
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                ANTHROPIC_API_URL, headers=headers, json=payload, timeout=45,
            )
            if resp.status_code == 429:
                wait = max(
                    int(resp.headers.get("retry-after", 2 ** (attempt + 1))),
                    2 ** (attempt + 1),
                )
                logger.info(f"Claude search rate-limited, waiting {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            parts = [
                b["text"] for b in data.get("content", []) if b.get("type") == "text"
            ]
            return "\n".join(parts).strip() if parts else None

        except Exception as e:
            logger.warning(f"Claude search call failed: {e}")
            return None

    logger.warning("Claude search: max retries exhausted")
    return None


# ═════════════════════════════════════════════════════════
#  GPT-5.4 (Azure AI Foundry — Responses API)
# ═════════════════════════════════════════════════════════

def _get_openai_client(api_key: Optional[str] = None):
    """Lazily build an OpenAI client pointed at Azure AI Foundry."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for GPT provider. "
            "Install with: pip install openai"
        )

    cfg = _get_azure_config()
    key = api_key or cfg["api_key"]
    endpoint = cfg["endpoint"].rstrip("/")

    if not key or not endpoint:
        raise ValueError(
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set "
            "to use the GPT provider."
        )

    # Azure AI Foundry exposes an OpenAI-compatible /openai/v1/ path
    base_url = (
        endpoint if endpoint.endswith("/openai/v1/")
        else f"{endpoint}/openai/v1/"
    )

    return OpenAI(api_key=key, base_url=base_url)


def _call_gpt(
    user_prompt: str,
    system_prompt: str = "",
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    use_web_search: bool = False,
) -> Optional[str]:
    """
    Call GPT via Azure AI Foundry Responses API.

    When *use_web_search* is True the model is given the web_search tool
    so it can ground its answer in live web results — this mirrors the
    Claude web_search_20250305 tool used by NewsEnricher.
    """
    cfg = _get_azure_config()
    deployment = model or cfg["deployment"]

    try:
        client = _get_openai_client(api_key)
    except (ImportError, ValueError) as e:
        logger.error(f"GPT provider setup failed: {e}")
        return None

    # Build tool list
    tools = []
    if use_web_search:
        tools.append({"type": "web_search"})

    # Build input: Responses API takes `input` (string or message list)
    # and `instructions` for system-level context.
    kwargs: dict = {
        "model": deployment,
        "input": user_prompt,
        "max_output_tokens": max_tokens,
    }
    # Reasoning / pro models reject temperature — only set it for
    # conventional (non-reasoning) models.
    if not _is_reasoning_model(deployment):
        kwargs["temperature"] = temperature
    if system_prompt:
        kwargs["instructions"] = system_prompt
    if tools:
        kwargs["tools"] = tools

    for attempt in range(max_retries):
        try:
            response = client.responses.create(**kwargs)

            # response.output_text is the convenience accessor that
            # concatenates all text output items.
            text = response.output_text
            if text:
                return text.strip()

            # Fallback: manually extract from output items
            parts = []
            for item in getattr(response, "output", []):
                if getattr(item, "type", "") == "message":
                    for block in getattr(item, "content", []):
                        if getattr(block, "type", "") == "output_text":
                            parts.append(block.text)
            return "\n".join(parts).strip() if parts else None

        except Exception as e:
            err_str = str(e)
            # Rate limiting — Azure returns 429
            if "429" in err_str or "rate" in err_str.lower():
                wait = 2 ** (attempt + 1)
                logger.info(f"GPT rate-limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            logger.error(f"GPT API call failed: {e}")
            return None

    logger.warning("GPT: max retries exhausted")
    return None
