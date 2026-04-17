from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .budget_guard import BudgetGuard
from .config import Settings
from .models import UsageRecord
from .retry import run_with_retry
from .tracing import TracingManager


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    return str(content)


def _extract_usage(response: Any, input_cost_per_1k: float, output_cost_per_1k: float) -> UsageRecord:
    usage_meta = getattr(response, "usage_metadata", {}) or {}
    response_meta = getattr(response, "response_metadata", {}) or {}
    token_usage = response_meta.get("token_usage", {}) or {}

    input_tokens = _to_int(
        usage_meta.get("input_tokens")
        or usage_meta.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or token_usage.get("prompt_tokens")
    )
    output_tokens = _to_int(
        usage_meta.get("output_tokens")
        or usage_meta.get("completion_tokens")
        or token_usage.get("output_tokens")
        or token_usage.get("completion_tokens")
    )
    total_tokens = _to_int(
        usage_meta.get("total_tokens")
        or token_usage.get("total_tokens")
        or (input_tokens + output_tokens)
    )

    estimated_cost_usd = (input_tokens / 1000.0) * input_cost_per_1k + (
        output_tokens / 1000.0
    ) * output_cost_per_1k

    return UsageRecord(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )


class OpenRouterClient:
    def __init__(self, settings: Settings, budget_guard: BudgetGuard, tracing: TracingManager) -> None:
        self.settings = settings
        self.budget_guard = budget_guard
        self.tracing = tracing

        provider = settings.llm_provider
        if provider not in {"openrouter", "openai"}:
            raise ValueError("LLM_PROVIDER must be either 'openrouter' or 'openai'.")

        if provider == "openai":
            api_key = settings.openai_api_key
            base_url = settings.openai_base_url
            model_name = settings.openai_model
        else:
            api_key = settings.openrouter_api_key
            base_url = settings.openrouter_base_url
            model_name = settings.openrouter_model

        if not api_key:
            expected = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
            raise ValueError(f"{expected} is required for provider '{provider}'.")

        self._api_key = api_key
        self._base_url = base_url
        self._default_model_name = model_name
        self._models: dict[str, ChatOpenAI] = {}

    def _build_model(self, model_name: str) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            model=model_name,
            temperature=self.settings.openrouter_temperature,
            max_tokens=self.settings.openrouter_max_tokens,
        )

    def _get_model(self, model_override: str | None) -> ChatOpenAI:
        selected = (model_override or "").strip() or self._default_model_name
        model = self._models.get(selected)
        if model is None:
            model = self._build_model(selected)
            self._models[selected] = model
        return model

    def invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        session_id: str,
        model_override: str | None = None,
    ) -> str:
        def _call() -> str:
            callback = self.tracing.build_callback_handler()
            config: dict[str, Any] = {
                "metadata": {"langfuse_session_id": session_id},
            }
            if callback is not None:
                config["callbacks"] = [callback]

            model = self._get_model(model_override)
            response = model.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
                config=config,
            )

            usage = _extract_usage(
                response=response,
                input_cost_per_1k=self.settings.token_cost_per_1k_input,
                output_cost_per_1k=self.settings.token_cost_per_1k_output,
            )
            self.budget_guard.consume(usage)

            return _normalize_text(response.content)

        return run_with_retry(
            operation=_call,
            max_retries=self.settings.max_retries,
            base_delay_seconds=self.settings.retry_base_delay_seconds,
        )
