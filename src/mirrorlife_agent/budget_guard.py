from __future__ import annotations

from dataclasses import dataclass

from .models import UsageRecord


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class BudgetSnapshot:
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class BudgetGuard:
    def __init__(self, max_usd: float, max_tokens: int) -> None:
        self.max_usd = max_usd
        self.max_tokens = max_tokens

        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._estimated_cost_usd = 0.0

    def consume(self, usage: UsageRecord) -> None:
        self._input_tokens += max(0, usage.input_tokens)
        self._output_tokens += max(0, usage.output_tokens)
        self._total_tokens += max(0, usage.total_tokens)
        self._estimated_cost_usd += max(0.0, usage.estimated_cost_usd)

        if self.max_tokens > 0 and self._total_tokens > self.max_tokens:
            raise BudgetExceededError(
                f"Token budget exceeded: {self._total_tokens} > {self.max_tokens}."
            )

        if self.max_usd > 0 and self._estimated_cost_usd > self.max_usd:
            raise BudgetExceededError(
                f"Estimated USD budget exceeded: {self._estimated_cost_usd:.6f} > {self.max_usd:.6f}."
            )

    def reset(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._estimated_cost_usd = 0.0

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            total_input_tokens=self._input_tokens,
            total_output_tokens=self._output_tokens,
            total_tokens=self._total_tokens,
            estimated_cost_usd=self._estimated_cost_usd,
        )
