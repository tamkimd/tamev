# decision_engine/data/quality_filter.py
from dataclasses import dataclass

from decision_engine.schema.decision import DecisionExample


@dataclass
class FilterResult:
    passed: bool
    reason: str
    to_hard_pool: bool = False


class DataQualityFilter:
    """
    Quality filtering pipeline:
    - Filters out excessively long contexts
    - Deduplicates examples by signature
    - Routes ambiguous / high-uncertainty teacher examples to hard-example pool
    """

    def __init__(self, min_teacher_confidence: float = 0.50, max_context_chars: int = 4000):
        self.min_teacher_confidence = min_teacher_confidence
        self.max_context_chars = max_context_chars
        self.seen_signatures: set[str] = set()

    def evaluate(self, example: DecisionExample) -> FilterResult:
        if len(example.state) > self.max_context_chars:
            return FilterResult(passed=False, reason="Context too long", to_hard_pool=False)

        sig = f"{example.state[:100]}||{example.question}"
        if sig in self.seen_signatures:
            return FilterResult(passed=False, reason="Duplicate example", to_hard_pool=False)
        self.seen_signatures.add(sig)

        if example.teacher_probs:
            max_p = max(example.teacher_probs.values())
            if max_p < self.min_teacher_confidence:
                return FilterResult(
                    passed=False, reason="Low teacher confidence", to_hard_pool=True
                )

        return FilterResult(passed=True, reason="OK", to_hard_pool=False)


# Convenience alias
QualityFilter = DataQualityFilter
