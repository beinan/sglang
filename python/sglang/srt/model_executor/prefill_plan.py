from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import nn

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class PrefillExecutionPlan(Protocol):
    """Cache-owned work executed eagerly within the model's forward context."""

    def execute(
        self, model: nn.Module, forward_batch: ForwardBatch, **kwargs
    ) -> LogitsProcessorOutput:
        """Complete cache writes before consuming KV and return real logits."""
        ...
