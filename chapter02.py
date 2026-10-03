"""Finite checks of aggregation, masking, collapse, risk and bounded planning.

Original teaching program; standard library only. No neural model is trained.
Real-valued inputs must be int/float (not bool), finite and representable as
binary64. Nonempty rectangular inputs and bounded actions are required.
"""
import cmath
import math


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be an int or float, not bool")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(name + " is not representable as binary64") from None
    if not math.isfinite(number):
        raise ValueError(name + " must be finite")
    return number


def vector(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name + " must be a nonempty list or tuple")
    return [finite(x, name) for x in values]


def matrix(rows, name):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError(name + " must have at least one row")
    result = [vector(row, name) for row in rows]
    if any(len(row) != len(result[0]) for row in result):
        raise ValueError(name + " must be rectangular")
    return result


def slot_updates(scores, values):
    """Inputs are score rows across slots and one scalar value per input.

    First normalize over slots; then normalize each slot over inputs.
    This is one aggregation step, without learned projections/GRU updates.
    """
    scores = matrix(scores, "scores")
    values = vector(values, "values")
    if len(scores) != len(values):
        raise ValueError("one value is required per input")
    weights = []
    for row in scores:
        maximum = max(row)
        exps = [math.exp(x - maximum) for x in row]
        total = math.fsum(exps)
        weights.append([x / total for x in exps])
    updates = []
    for slot in range(len(scores[0])):
        mass = math.fsum(row[slot] for row in weights)
        if mass == 0:
            raise ValueError("a slot has zero representable assignment mass")
        terms = [row[slot] / mass * val for row, val in zip(weights, values)]
        try:
            update = math.fsum(terms)
        except OverflowError:
            raise ValueError("slot update is not representable") from None
        updates.append(finite(update, "slot update"))
    return weights, updates


def anchored_tokens(history, masked, scale, times):
    """Scalar reduction of vector object slots; final row is masked future."""
    history = matrix(history, "history")
    scale = finite(scale, "anchor scale")
    times = vector(times, "time codes")
    if len(times) != len(history) + 1:
        raise ValueError("one time code per observed frame plus future is required")
    count = len(history[0])
    if not isinstance(masked, (set, frozenset)):
        raise ValueError("masked must be a set of slot indices")
    if any(type(i) is not int or not 0 <= i < count for i in masked):
        raise ValueError("masked slot index is invalid")
    result = [history[0][:]]
    for t in range(1, len(times)):
        row = []
        for i in range(count):
            value = (scale * history[0][i] + times[t]
                     if t == len(history) or i in masked else history[t][i])
            row.append(finite(value, "anchored token"))
        result.append(row)
    return result


def cf_discrepancy(samples, frequencies):
    """Average finite-frequency squared discrepancy from standard Gaussian CF.

    Not the integral Epps-Pulley statistic and not a normality certificate.
    """
    samples = vector(samples, "samples")
    frequencies = vector(frequencies, "frequencies")
    if any(abs(x) > 1e150 for x in samples + frequencies):
        raise ValueError("CF fixture magnitudes must not exceed 1e150")
    errors = []
    for frequency in frequencies:
        empirical = sum(cmath.exp(1j * frequency * x) for x in samples) / len(samples)
        target = math.exp(-0.5 * frequency * frequency)
        errors.append(abs(empirical - target) ** 2)
    return math.fsum(errors) / len(errors)


def prediction_loss(predicted, targets):
    predicted = vector(predicted, "predicted")
    targets = vector(targets, "targets")
    if len(predicted) != len(targets):
        raise ValueError("prediction and target lengths must match")
    if any(abs(x) > 1e100 for x in predicted + targets):
        raise ValueError("loss fixture magnitudes must not exceed 1e100")
    return math.fsum((a-b)**2 for a, b in zip(predicted, targets)) / len(targets)


def scalar_rollout(start, actions, bound):
    """Known additive dynamics x_next=x+a; check every physical action."""
    start = finite(start, "start")
    actions = vector(actions, "actions")
    bound = finite(bound, "action bound")
    if bound <= 0:
        raise ValueError("action bound must be positive")
    if any(abs(a) > bound for a in actions):
        raise ValueError("action exceeds the permitted bound")
    states = [start]
    for action in actions:
        states.append(finite(states[-1] + action, "state"))
    return states


def plan(start, goal, horizon, bound, effort):
    """Solve the stated strictly convex additive toy by its closed form.

    With identical bounds, the optimum distributes total displacement evenly.
    A positive effort coefficient gives a unique action sequence.
    """
    start, goal = finite(start, "start"), finite(goal, "goal")
    bound, effort = finite(bound, "bound"), finite(effort, "effort")
    if type(horizon) is not int or not 1 <= horizon <= 10000:
        raise ValueError("horizon must be an integer from 1 to 10000")
    if bound <= 0 or effort <= 0:
        raise ValueError("bound and effort must be positive")
    if max(abs(start), abs(goal), bound, effort) > 1e100:
        raise ValueError("planning fixture magnitudes must not exceed 1e100")
    action = max(-bound, min(bound, (goal - start) / (horizon + effort)))
    actions = [action] * horizon
    states = scalar_rollout(start, actions, bound)
    cost = (states[-1] - goal) ** 2 + effort * math.fsum(a*a for a in actions)
    return actions, states, finite(cost, "cost")


def expect_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("invalid input was accepted")


def main():
    scores = [[math.log(3), 0], [0, math.log(3)]]
    weights, updates = slot_updates(scores, [2, 10])
    assert all(math.isclose(math.fsum(row), 1) for row in weights)
    assert all(math.isclose(a, b) for a, b in zip(updates, [4, 8]))
    _, three_updates = slot_updates(scores + [[0, 0]], [2, 10, 6])
    assert all(math.isclose(a, b) for a, b in zip(three_updates, [14/3, 22/3]))
    print("slot updates:", " ".join(f"{x:.6f}" for x in updates))
    tokens = anchored_tokens([[1, 10], [2, 20], [3, 30]], {0}, 0.5,
                             [0, 0.1, 0.2, 0.3])
    assert tokens == [[1.0, 10.0], [0.6, 20.0], [0.7, 30.0], [0.8, 5.3]]
    swapped = anchored_tokens([[10, 1], [20, 2], [30, 3]], {1}, 0.5,
                              [0, 0.1, 0.2, 0.3])
    assert swapped == [list(reversed(row)) for row in tokens]
    print("masked slot 0:", " ".join(f"{row[0]:.6f}" for row in tokens))
    print("future slot 1:", f"{tokens[-1][1]:.6f}")
    target = [-1.0, 1.0, -2.0, 2.0]
    best_mean_risk = math.fsum(y*y for y in target) / len(target)
    assert best_mean_risk == 2.5 and sum(target) == 0
    additive_target = [0, 2, 1, 3]
    assert prediction_loss([1, 1, 2, 2], additive_target) == 1
    assert prediction_loss([1.5]*4, additive_target) == 1.25
    print("risk with/without X:", f"{best_mean_risk:.6f}",
          f"{best_mean_risk:.6f}")
    constant_encoder = lambda observation: 0.0
    constant_predictor = lambda latent: 0.0
    encodings = [constant_encoder(y) for y in target]
    predictions = [constant_predictor(z) for z in encodings]
    zero_prediction_loss = prediction_loss(predictions, encodings)
    assert zero_prediction_loss == 0 and prediction_loss([1, 3], [2, 1]) == 2.5
    print("collapsed prediction loss:", f"{zero_prediction_loss:.6f}")
    print("collapsed CF discrepancy:", f"{cf_discrepancy([0, 0], [1]):.6f}")
    print("two-point CF discrepancy:", f"{cf_discrepancy([-1, 1], [1]):.6f}")
    actions, states, cost = plan(0, 3, 2, 1, 0.25)
    assert actions == [1.0, 1.0] and states == [0.0, 1.0, 2.0]
    assert cost == 1.5
    print("bounded actions:", " ".join(f"a={x:.6f}" for x in actions))
    print("terminal state/cost:", f"{states[-1]:.6f}", f"{cost:.6f}")
    interior, _, _ = plan(0, 1, 2, 1, 0.25)
    assert all(math.isclose(x, 4/9) for x in interior)
    _, _, interior_cost = plan(0, 1, 2, 1, 0.25)
    assert math.isclose(interior_cost, 1/9)
    _, one_step, _ = plan(0, 3, 1, 1, 0.25)
    assert one_step[-1] == 1.0
    expect_error(slot_updates, [[0], [0, 1]], [1, 2])
    expect_error(slot_updates, [[10**1000, 0]], [1])
    expect_error(slot_updates, [[True, 0]], [1])
    expect_error(slot_updates, [[math.inf]], [1])
    expect_error(anchored_tokens, [[1]], {1}, 0.5, [0, 1])
    expect_error(anchored_tokens, [[1]], {0}, 0.5, [0])
    expect_error(cf_discrepancy, [], [1])
    expect_error(cf_discrepancy, [1e151], [1])
    expect_error(prediction_loss, [1], [1, 2])
    expect_error(prediction_loss, [math.nan], [0])
    expect_error(scalar_rollout, 0, [1.1], 1)
    expect_error(scalar_rollout, 1e308, [1e308], 1e308)
    expect_error(plan, 0, 3, True, 1, 0.25)
    expect_error(plan, 0, 3, 0, 1, 0.25)
    expect_error(plan, 0, 3, 2, 1, 0)
    expect_error(plan, 0, 10**1000, 2, 1, 0.25)
    print("checks: passed")


if __name__ == "__main__":
    main()
