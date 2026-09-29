# decision_engine/training/hard_mining.py

from decision_engine.schema.decision import DecisionExample


def mine_uncertain_examples(
    examples: list[DecisionExample], uncertainty_margin: float = 0.2
) -> list[DecisionExample]:
    """
    Selects examples where the top-2 predicted/teacher probabilities have margin < uncertainty_margin.
    These examples represent high model ambiguity and are routed to Teacher active review.
    """
    hard_examples = []
    for ex in examples:
        if not ex.teacher_probs or len(ex.teacher_probs) < 2:
            continue
        sorted_probs = sorted(ex.teacher_probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1]
        if margin < uncertainty_margin:
            hard_examples.append(ex)
    return hard_examples
