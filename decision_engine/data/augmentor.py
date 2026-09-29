# decision_engine/data/augmentor.py
import random

from decision_engine.schema.decision import DecisionExample


def permute_options(
    example: DecisionExample, permutation: list[int] | None = None
) -> tuple[DecisionExample, list[int]]:
    """
    Shuffles option positions while preserving the correct option semantic assignment.
    Ensures the model learns option semantics and compatibility rather than positional bias.
    """
    if example.decision_type.value != "choice" or not example.options:
        return example, []

    k = len(example.options)
    if permutation is None:
        permutation = list(range(k))
        random.shuffle(permutation)

    shuffled_options = [example.options[i] for i in permutation]

    new_data = example.to_dict()
    new_data["options"] = [opt.to_dict() for opt in shuffled_options]

    permuted_example = DecisionExample.from_dict(new_data)
    return permuted_example, permutation
