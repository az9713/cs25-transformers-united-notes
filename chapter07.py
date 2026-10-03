"""Original scientific-decision fixtures; no biological predictor or LLM service.

Inputs are finite int/float values converted to binary64; bool is excluded.
All outputs are fictional teaching quantities. Independence is an assumption,
not established by running this program. Tiny final probabilities may round to 0.
"""
import math


def finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be an int or float, excluding bool")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(name + " is not representable") from exc
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def probability(value, name):
    value = finite_real(value, name)
    if not 0 <= value <= 1:
        raise ValueError(name + " must lie in [0, 1]")
    return value


def logistic(log_odds):
    log_odds = finite_real(log_odds, "log_odds")
    if log_odds >= 0:
        return 1 / (1 + math.exp(-log_odds))
    x = math.exp(log_odds)
    return x / (1 + x)


def elo_pair(rating_i, rating_j, outcome, step=32.0):
    """Stipulated scale-400 Elo, not a claim about the source implementation."""
    rating_i = finite_real(rating_i, "rating_i")
    rating_j = finite_real(rating_j, "rating_j")
    if max(abs(rating_i), abs(rating_j)) > 1e6:
        raise ValueError("ratings must lie in [-1e6, 1e6]")
    outcome = probability(outcome, "outcome")
    if outcome not in (0, 0.5, 1):
        raise ValueError("outcome must be a loss, tie or win")
    step = finite_real(step, "step")
    if not 0 <= step <= 1000:
        raise ValueError("step must lie in [0, 1000]")
    expected = logistic((rating_i - rating_j) * math.log(10) / 400)
    increment = step * (outcome - expected)
    return rating_i + increment, rating_j - increment, expected


def posterior_positive(prior, sensitivity, false_positive):
    """Binary Bayes model; exhaustive alternatives and known likelihoods required."""
    prior = probability(prior, "prior")
    sensitivity = probability(sensitivity, "sensitivity")
    false_positive = probability(false_positive, "false_positive")
    possible_h = prior > 0 and sensitivity > 0
    possible_other = prior < 1 and false_positive > 0
    if not possible_h and not possible_other:
        raise ValueError("positive evidence has zero probability")
    if not possible_h:
        return 0.0
    if not possible_other:
        return 1.0
    # Log weights preserve likelihoods when direct multiplication underflows.
    log_h = math.log(prior) + math.log(sensitivity)
    log_other = math.log1p(-prior) + math.log(false_positive)
    return logistic(log_h - log_other)


def posterior_independent_records(prior, records):
    """Each (unique ID, sensitivity, false-positive) is assumed conditionally independent.

    Unique IDs prevent counting the same record twice. Distinct IDs do not
    establish independence: the caller must justify the statistical model.
    """
    prior = probability(prior, "prior")
    if not isinstance(records, (list, tuple)) or len(records) > 10000:
        raise ValueError("records must be a list/tuple of at most 10000 entries")
    log_h = math.log(prior) if prior > 0 else -math.inf
    log_other = math.log1p(-prior) if prior < 1 else -math.inf
    seen = set()
    for record in records:
        if not isinstance(record, (tuple, list)) or len(record) != 3:
            raise ValueError("records require ID and two likelihoods")
        ident, sensitivity, false_positive = record
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("record IDs must be nonempty, unique strings")
        seen.add(ident)
        sensitivity = probability(sensitivity, "sensitivity")
        false_positive = probability(false_positive, "false_positive")
        log_h += math.log(sensitivity) if sensitivity > 0 else -math.inf
        log_other += math.log(false_positive) if false_positive > 0 else -math.inf
        if log_h == log_other == -math.inf:
            raise ValueError("evidence sequence has zero probability")
    if log_h == -math.inf:
        return 0.0
    if log_other == -math.inf:
        return 1.0
    return logistic(log_h - log_other)


def experiment_value(prior, sensitivity, false_positive, cost):
    """Toy action: utility 10p-5, abstention 0; conditional assay model."""
    prior = probability(prior, "prior")
    sensitivity = probability(sensitivity, "sensitivity")
    false_positive = probability(false_positive, "false_positive")
    cost = finite_real(cost, "cost")
    if not 0 <= cost <= 1000:
        raise ValueError("cost must lie in [0, 1000] fictional utility units")
    q = prior * sensitivity + (1 - prior) * false_positive
    value = 0.0
    if q > 0:
        plus = posterior_positive(prior, sensitivity, false_positive)
        value += q * max(0.0, 10 * plus - 5)
    if q < 1:
        minus = posterior_positive(prior, 1 - sensitivity, 1 - false_positive)
        value += (1 - q) * max(0.0, 10 * minus - 5)
    before = max(0.0, 10 * prior - 5)
    return value - before, value - before - cost


def false_any(tests, alpha):
    """Independent null tests only; this is not a correlated-test estimate."""
    if isinstance(tests, bool) or not isinstance(tests, int) or not 1 <= tests <= 10**6:
        raise ValueError("tests must be an integer in [1, 1e6]")
    alpha = probability(alpha, "alpha")
    if alpha == 1:
        return 1.0
    return -math.expm1(tests * math.log1p(-alpha))


def mean_variance_ratio(reviews, correlation):
    if isinstance(reviews, bool) or not isinstance(reviews, int) or not 2 <= reviews <= 10**6:
        raise ValueError("reviews must be an integer in [2, 1e6]")
    correlation = finite_real(correlation, "correlation")
    if not -1 / (reviews - 1) <= correlation <= 1:
        raise ValueError("correlation violates the equicorrelation PSD bound")
    return (1 + (reviews - 1) * correlation) / reviews


def hill_viability(concentration, midpoint, exponent, lower=0.0, upper=1.0):
    concentration = finite_real(concentration, "concentration")
    midpoint = finite_real(midpoint, "midpoint")
    exponent = finite_real(exponent, "exponent")
    lower = probability(lower, "lower")
    upper = probability(upper, "upper")
    if concentration < 0 or midpoint <= 0 or not 0 < exponent <= 100 or lower > upper:
        raise ValueError("require c>=0, midpoint>0, 0<h<=100 and lower<=upper")
    if concentration == 0:
        return upper
    z = exponent * (math.log(concentration) - math.log(midpoint))
    return lower + (upper - lower) * logistic(-z)


def per_viable_cell(marker, viable_cells):
    marker = finite_real(marker, "marker")
    viable_cells = finite_real(viable_cells, "viable_cells")
    if marker < 0 or viable_cells <= 0:
        raise ValueError("require nonnegative marker and positive viable-cell count")
    result = marker / viable_cells
    if not math.isfinite(result) or (marker > 0 and result == 0):
        raise ValueError("positive per-cell ratio is not representable")
    return result


def main():
    a, b, _ = elo_pair(1200, 1200, 1)
    a, b, expected = elo_pair(a, b, 1)
    assert math.isclose(a + b, 2400)
    print(f"two Elo wins: {a:.6f}, {b:.6f}; second expectation {expected:.6f}")
    for _ in range(18):
        a, b, _ = elo_pair(a, b, 1)
    # Reference labels are stipulated; Elo sees only preference outcomes.
    assert a > b
    print(f"20 wins for stipulated false proposal: {a:.6f} > {b:.6f}")
    one = posterior_positive(0.1, 0.8, 0.1)
    two = posterior_independent_records(0.1, [('assay-1', .8, .1), ('assay-2', .8, .1)])
    assert math.isclose(one, 8 / 17) and math.isclose(two, 64 / 73)
    print(f"one positive: {one:.6f}; two conditionally independent: {two:.6f}")
    gross, net = experiment_value(.2, .8, .1, .3)
    assert math.isclose(gross, .4) and math.isclose(net, .1)
    print(f"experiment value: gross {gross:.6f}; net {net:.6f}")
    ratio = mean_variance_ratio(10, .8)
    print(f"10 correlated reviews: variance ratio {ratio:.6f}; effective count {1/ratio:.6f}")
    print(f"100 independent null tests: false-any {false_any(100, .05):.6f}")
    absolute_half = 10 * math.sqrt(5 / 3)
    assert math.isclose(hill_viability(absolute_half, 10, 2, .2, 1), .5)
    print(f"Hill midpoint viability {hill_viability(10,10,2,.2,1):.6f}; absolute half at {absolute_half:.6f}")
    print(f"same marker 50: per-cell {per_viable_cell(50,100):.6f} or {per_viable_cell(50,50):.6f}")
    assert math.isclose(posterior_positive(1e-300, 1e-300, 1e-300), 1e-300, rel_tol=1e-12, abs_tol=0)
    assert hill_viability(1e300, 1e-300, 100) == 0
    assert hill_viability(1e-300, 1e300, 100) == 1
    assert math.isclose(false_any(10**6, 1e-300), 1e-294, rel_tol=1e-12, abs_tol=0)
    assert experiment_value(.2, 0, 0, 0) == (0.0, 0.0)
    assert experiment_value(.2, 1, 1, 0) == (0.0, 0.0)
    assert posterior_positive(0, .8, .1) == 0 and posterior_positive(1, .8, .1) == 1
    balanced = posterior_independent_records(.5, [('a', 1, 1e-300), ('b', 1e-300, 1)])
    assert math.isclose(balanced, .5)
    assert logistic(0) == .5
    assert logistic(1e308) == 1
    assert logistic(-1e308) == 0
    assert per_viable_cell(0, 1e308) == 0
    assert per_viable_cell(5e-324, 1) == 5e-324
    print("14 extreme/endpoint assertions passed")
    bad = [
        lambda: finite_real(True, 'x'), lambda: finite_real(10**1000, 'x'),
        lambda: finite_real(float('nan'), 'x'), lambda: finite_real('1', 'x'),
        lambda: elo_pair(float('inf'), 1200, 1), lambda: elo_pair(1e7, 1200, 1),
        lambda: elo_pair(1200, 1200, .7), lambda: elo_pair(1200, 1200, 1, -1),
        lambda: posterior_positive(-.1, .8, .1), lambda: posterior_positive(.1, 0, 0),
        lambda: posterior_positive(0, 1, 0), lambda: posterior_positive(.1, 2, .1),
        lambda: posterior_independent_records(.1, [('x', .8, .1), ('x', .8, .1)]),
        lambda: posterior_independent_records(.1, [('x', .8)]),
        lambda: posterior_independent_records(.1, None),
        lambda: experiment_value(.2, .8, .1, -1), lambda: false_any(True, .05),
        lambda: false_any(0, .05), lambda: false_any(2, -1),
        lambda: mean_variance_ratio(1, .8), lambda: mean_variance_ratio(3, -.6),
        lambda: hill_viability(-1, 10, 2), lambda: hill_viability(1, 0, 2),
        lambda: hill_viability(1, 10, 0), lambda: hill_viability(1, 10, 2, .8, .2),
        lambda: per_viable_cell(1, 0), lambda: per_viable_cell(1e308, 1e-300),
        lambda: per_viable_cell(5e-324, 1e308),
        lambda: logistic(float('nan')), lambda: logistic(True),
        lambda: logistic(10**1000), lambda: logistic(float('inf')),
        lambda: logistic(float('-inf')),
    ]
    for case in bad:
        try:
            case()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    print(f"{len(bad)} invalid-input checks passed")


if __name__ == '__main__':
    main()
