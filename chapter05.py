"""Finite curriculum/reward/compute checks; no model, files, network or randomness."""
import math
from fractions import Fraction


def finite_real(value, name="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be an int or float, excluding bool")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(name + " is not representable") from None
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def vector(values, name="vector"):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name + " must be a nonempty list or tuple")
    return [finite_real(x, name) for x in values]


def distribution(values, name="distribution", strictly_positive=False):
    result = vector(values, name)
    if any(x < 0 or x > 1 or (strictly_positive and x == 0) for x in result):
        raise ValueError(name + " has an invalid probability")
    if not math.isclose(math.fsum(result), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError(name + " must sum to one")
    return result


def _scaled_product_ratio(left, right, divisor):
    """Nonnegative product/positive divisor without destructive intermediates."""
    if left == 0 or right == 0:
        return 0.0
    lm, le = math.frexp(left)
    rm, re = math.frexp(right)
    dm, de = math.frexp(divisor)
    try:
        result = finite_real(math.ldexp(lm * rm / dm, le + re - de), "ratio")
    except OverflowError:
        raise ValueError("ratio is not representable") from None
    if result == 0:
        raise ValueError("positive ratio underflows to zero")
    return result


def exposures(total_tokens, unique_tokens, weights):
    total = finite_real(total_tokens, "total tokens")
    sizes = vector(unique_tokens, "unique tokens")
    blend = distribution(weights, "weights")
    if total <= 0 or any(x <= 0 for x in sizes) or len(sizes) != len(blend):
        raise ValueError("positive budget/source sizes and matching dimensions required")
    return [_scaled_product_ratio(total, weight, size)
            for size, weight in zip(sizes, blend)]


def scalar_train(targets, learning_rate, initial=0):
    targets = vector(targets, "targets")
    rate = finite_real(learning_rate, "learning rate")
    state = finite_real(initial, "initial state")
    if not 0 < rate <= 1:
        raise ValueError("learning rate must lie in (0,1]")
    for target in targets:
        state = finite_real((1-rate)*state + rate*target, "updated state")
    return state


def log_gain(reasoned_probability, baseline_probability):
    p = finite_real(reasoned_probability, "reasoned probability")
    b = finite_real(baseline_probability, "baseline probability")
    if not 0 < p <= 1 or not 0 < b <= 1:
        raise ValueError("scored probabilities must lie in (0,1]")
    return math.log(p) - math.log(b)


def corrected_advantages(rewards):
    rewards = vector(rewards, "rewards")
    count = len(rewards)
    if count < 2:
        raise ValueError("at least two rewards required")
    # Binary64 inputs have exact rational representations. Accumulate before
    # rounding, so large cancellations cannot discard a tiny, finite residual.
    anchor = Fraction(rewards[0])
    differences = [Fraction(x)-anchor for x in rewards]
    mean = sum(differences, Fraction(0)) / count
    try:
        rounded_mean = finite_real(float(mean), "centered mean")
    except OverflowError:
        raise ValueError("centered mean is not representable") from None
    if mean and rounded_mean == 0:
        raise ValueError("nonzero centered mean underflows to zero")
    result = []
    for difference in differences:
        exact = (difference-mean) * count / (count-1)
        try:
            rounded = finite_real(float(exact), "advantage")
        except OverflowError:
            raise ValueError("advantage is not representable") from None
        if exact and rounded == 0:
            raise ValueError("nonzero advantage underflows to zero")
        result.append(rounded)
    return result


def clipped_objective(ratio, advantage, lower_width=0.2, upper_width=0.2):
    ratio = finite_real(ratio, "importance ratio")
    advantage = finite_real(advantage, "advantage")
    low = finite_real(lower_width, "lower clipping width")
    high = finite_real(upper_width, "upper clipping width")
    if ratio <= 0 or not 0 <= low < 1 or high < 0:
        raise ValueError("positive ratio and valid clipping widths required")
    upper = finite_real(1+high, "upper clipping bound")
    clipped = max(1-low, min(upper, ratio))
    raw = finite_real(ratio*advantage, "raw surrogate")
    bounded = finite_real(clipped*advantage, "clipped surrogate")
    return min(raw, bounded)


def cross_entropy(data_distribution, predictor):
    truth = distribution(data_distribution, "data distribution")
    prediction = distribution(predictor, "predictor", strictly_positive=True)
    if len(truth) != len(prediction):
        raise ValueError("distribution dimensions must match")
    return math.fsum(-a*math.log(b) for a, b in zip(truth, prediction))


def marginal_bound(thought_weights, target_probabilities, baseline_probability):
    weights = distribution(thought_weights, "thought weights")
    probabilities = vector(target_probabilities, "target probabilities")
    if len(weights) != len(probabilities):
        raise ValueError("thought dimensions must match")
    rewards = [log_gain(p, baseline_probability) for p in probabilities]
    expected = math.fsum(w*r for w, r in zip(weights, rewards))
    marginal = math.fsum(w*p for w, p in zip(weights, probabilities))
    return expected, log_gain(marginal, baseline_probability)


def compute_tokens(documents, prefix_length, group_size, thought_length):
    """Idealized counts, not an implementation of the paper's FLOP estimator."""
    values = (documents, prefix_length, group_size, thought_length)
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in values):
        raise ValueError("compute dimensions must be positive integers")
    if group_size < 2:
        raise ValueError("group size must be at least two")
    input_tokens = documents*prefix_length
    generated = documents*group_size*thought_length
    # Original teaching approximation: train thoughts at 6P and generate at 2P.
    try:
        equivalent = generated + generated/3
    except OverflowError:
        raise ValueError("equivalent token count is not representable") from None
    if not math.isfinite(equivalent):
        raise ValueError("equivalent token count is not representable")
    return input_tokens, generated, equivalent


def check():
    epochs = exposures(1000, [800, 100, 100], [.6, .3, .1])
    assert epochs == [.75, 3, 1]
    # Finite result despite an otherwise overflowing multiplication.
    assert exposures(1e308, [1e308], [1]) == [1]
    assert scalar_train([0, 2], .25) == .5
    assert scalar_train([2, 0], .25) == .375
    rewards = [log_gain(p, .2) for p in [.6, .2, .1]]
    advantages = corrected_advantages(rewards)
    shifted = corrected_advantages([log_gain(p, .4) for p in [.6, .2, .1]])
    assert all(math.isclose(a,b,abs_tol=1e-14) for a,b in zip(advantages,shifted))
    assert math.isclose(math.fsum(advantages), 0, abs_tol=1e-14)
    assert corrected_advantages([1e308,1e308]) == [0,0]
    assert corrected_advantages([5e-324,5e-324]) == [0,0]
    cancellation = corrected_advantages([0, 1e308, -1e308, 1e-300])
    assert math.isclose(cancellation[0], -1e-300/3, rel_tol=1e-14, abs_tol=0)
    assert math.isclose(cancellation[3], 1e-300, rel_tol=1e-14, abs_tol=0)
    assert corrected_advantages([-.7,-.9])[0] > 0
    assert corrected_advantages([2,2]) == [0,0]
    assert math.isclose(clipped_objective(1.5,2),2.4)
    assert math.isclose(clipped_objective(.5,-2),-1.6)
    expected, bound = marginal_bound([.5,.5],[.6,.1],.2)
    assert expected < bound
    assert math.isclose(cross_entropy([.75,.25],[.5,.5]) -
                        cross_entropy([.75,.25],[.75,.25]), .130812035941137)
    assert compute_tokens(100,100,4,20) == (10000,8000,8000+8000/3)
    invalids = [
        lambda: finite_real(True), lambda: finite_real(10**1000),
        lambda: finite_real(math.nan), lambda: finite_real(".2"),
        lambda: exposures(0,[1],[1]), lambda: exposures(1,[0],[1]),
        lambda: exposures(1,[1,2],[1]), lambda: exposures(1,[1],[.5]),
        lambda: exposures(1e308,[1e-308],[1]),
        lambda: exposures(1e-308,[1e308],[1]),
        lambda: scalar_train([], .2), lambda: scalar_train([1],0),
        lambda: log_gain(0,.2), lambda: log_gain(1.1,.2),
        lambda: corrected_advantages([1]),
        lambda: corrected_advantages([1e308,-1e308]),
        lambda: corrected_advantages([5e-324,1e-323]),
        lambda: clipped_objective(0,1), lambda: clipped_objective(1,1,1),
        lambda: cross_entropy([1,0],[1,0]),
        lambda: marginal_bound([1],[.2,.3],.2),
        lambda: compute_tokens(True,2,2,2), lambda: compute_tokens(1,2,1,2),
        lambda: compute_tokens(10**1000,2,2,2),
    ]
    for call in invalids:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    print("expected source epochs:", epochs)
    print("curriculum endpoints: %.6f / %.6f" %
          (scalar_train([0,2],.25),scalar_train([2,0],.25)))
    print("log-gain rewards:", [round(x,6) for x in rewards])
    print("corrected advantages:", [round(x,6) for x in advantages])
    print("common-baseline cancellation: passed")
    print("expected reward / marginal bound: %.6f / %.6f" % (expected,bound))
    print("signed clipped objectives: %.6f / %.6f" %
          (clipped_objective(1.5,2),clipped_objective(.5,-2)))
    print("input / generated / training-equivalent tokens: %d / %d / %.6f" %
          compute_tokens(100,100,4,20))
    print("All finite mathematical checks and %d invalid-input checks passed." % len(invalids))


if __name__ == "__main__":
    check()
