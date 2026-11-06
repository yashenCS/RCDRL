def constraint_metrics(success_rate: float, reference_success_rate: float, relative_deadline_gap: float) -> dict:

    success_rate = float(success_rate)
    reference_success_rate = float(reference_success_rate)
    relative_deadline_gap = float(relative_deadline_gap)
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError(f"success_rate must lie in [0, 1], but got {success_rate}.")
    if not 0.0 <= reference_success_rate <= 1.0:
        raise ValueError(
            f"reference_success_rate must lie in [0, 1], but got {reference_success_rate}."
        )
    if not 0.0 <= relative_deadline_gap < 1.0:
        raise ValueError(
            f"relative_deadline_gap must lie in [0, 1), but got {relative_deadline_gap}."
        )

    minimum_success_rate = (1.0 - relative_deadline_gap) * reference_success_rate
    deadline_gap = (
        (reference_success_rate - success_rate) / reference_success_rate
        if reference_success_rate > 0.0
        else 0.0
    )
    constraint_violation = max(0.0, minimum_success_rate - success_rate)
    return {
        "reference_success_rate": reference_success_rate,
        "minimum_success_rate": minimum_success_rate,
        "maximum_miss_rate": 1.0 - minimum_success_rate,
        "relative_deadline_gap": relative_deadline_gap,
        "deadline_gap": deadline_gap,
        "constraint_violation": constraint_violation,
        "feasible": bool(success_rate + 1e-12 >= minimum_success_rate),
    }


def candidate_rank(candidate: dict) -> tuple:

    model_num = int(candidate["model_num"])
    vm_cost = float(candidate["mean_vm_cost_per_episode"])
    if bool(candidate["feasible"]):
        return (0, vm_cost, model_num)
    return (1, float(candidate["constraint_violation"]), vm_cost, model_num)


def choose_better_candidate(current: dict, candidate: dict) -> dict:

    if current is None or candidate_rank(candidate) < candidate_rank(current):
        return candidate
    return current
