"""Original finite checks for recurrence, routing and likelihood units."""
import math


def finite_real(value, name="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + " must be a real int or float")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " is not representable") from error
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def real_sequence(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name + " must be a nonempty list or tuple")
    return [finite_real(value, name) for value in values]


def coefficients(transitions, injections):
    a = real_sequence(transitions, "transition")
    r = real_sequence(injections, "injection")
    if len(a) != len(r):
        raise ValueError("coefficient lengths must agree")
    return a, r


def compose(earlier, later):
    """Chronological affine composition; later acts after earlier."""
    if not all(isinstance(pair, (list, tuple)) and len(pair) == 2
               for pair in (earlier, later)):
        raise ValueError("each affine pair must have two entries")
    a, r = (finite_real(value) for value in earlier)
    d, s = (finite_real(value) for value in later)
    return finite_real(d*a, "transition product"), finite_real(d*r+s)


def serial_states(transitions, injections, initial=0):
    a, r = coefficients(transitions, injections)
    state = finite_real(initial, "initial state")
    outputs = []
    for transition, injection in zip(a, r):
        state = finite_real(transition*state + injection, "state")
        outputs.append(state)
    return outputs


def scan_states(transitions, injections, initial=0):
    """Blelloch-style tree: O(n) composition work, serial Python loops."""
    a, r = coefficients(transitions, injections)
    initial = finite_real(initial, "initial state")
    pairs = list(zip(a, r))
    size = 1
    while size < len(pairs):
        size *= 2
    tree = pairs + [(1.0, 0.0)]*(size - len(pairs))
    stride = 2
    while stride <= size:
        for right in range(stride - 1, size, stride):
            left = right - stride//2
            tree[right] = compose(tree[left], tree[right])
        stride *= 2
    tree[-1] = (1.0, 0.0)
    stride = size
    while stride >= 2:
        for right in range(stride - 1, size, stride):
            left = right - stride//2
            old_left, prefix = tree[left], tree[right]
            tree[left] = prefix
            tree[right] = compose(prefix, old_left)
        stride //= 2
    outputs = []
    for prefix, pair in zip(tree, pairs):
        transition, injection = compose(prefix, pair)
        outputs.append(finite_real(transition*initial + injection))
    return outputs


def unrolled_states(transitions, injections, initial=0):
    """Independent quadratic expansion; products never divide by gates."""
    a, r = coefficients(transitions, injections)
    initial = finite_real(initial, "initial state")
    outputs = []
    for last in range(len(a)):
        contribution = 0.0
        product = 1.0
        for first in range(last, -1, -1):
            contribution = finite_real(contribution + product*r[first])
            product = finite_real(product*a[first])
        outputs.append(finite_real(contribution + product*initial))
    return outputs


def zoh_coefficients(decay_rate, step, coupling):
    """Scalar constant-input formula, including underflowed interval exponents.

    Use a limiting ratio near zero and binary scaling for the gain products.
    A final unrepresentable coefficient or nonfinite exponent is rejected.
    """
    rate = finite_real(decay_rate, "decay rate")
    step = finite_real(step, "step")
    coupling = finite_real(coupling, "coupling")
    if rate <= 0 or step < 0:
        raise ValueError("positive rate and nonnegative step required")
    exponent = finite_real(rate*step, "interval exponent")
    transition = math.exp(-exponent)
    if step == 0 or coupling == 0:
        return transition, 0.0
    if exponent <= 1:
        ratio = 1.0 if exponent == 0 else -math.expm1(-exponent)/exponent
        c_mantissa, c_power = math.frexp(coupling)
        s_mantissa, s_power = math.frexp(step)
        r_mantissa, r_power = math.frexp(ratio)
        mantissa = c_mantissa*s_mantissa*r_mantissa
        power = c_power+s_power+r_power
    else:
        c_mantissa, c_power = math.frexp(coupling)
        f_mantissa, f_power = math.frexp(-math.expm1(-exponent))
        d_mantissa, d_power = math.frexp(rate)
        mantissa = c_mantissa*f_mantissa/d_mantissa
        power = c_power+f_power-d_power
    try:
        gain = finite_real(math.ldexp(mantissa, power), "injection gain")
    except OverflowError:
        raise ValueError("injection gain is not representable") from None
    return transition, gain


def causal_expand(values, scores):
    """Identity main network, selected EMA, latest-boundary expansion."""
    values = real_sequence(values, "encoder value")
    scores = real_sequence(scores, "boundary score")
    if len(values) != len(scores):
        raise ValueError("value and score lengths must agree")
    if scores[0] != 1 or any(not 0 <= score <= 1 for score in scores):
        raise ValueError("scores in [0, 1], first score exactly 1")
    latest = 0.0
    selected = []
    expanded = []
    for position, (value, score) in enumerate(zip(values, scores)):
        if score >= 0.5:
            latest = finite_real(score*value + (1-score)*latest)
            selected.append(position)
        expanded.append(latest)
    return selected, expanded


def bits_per_byte(probabilities, byte_count):
    probabilities = real_sequence(probabilities, "probability")
    if (isinstance(byte_count, bool) or not isinstance(byte_count, int)
            or byte_count <= 0):
        raise ValueError("byte count must be a positive integer")
    count = finite_real(byte_count, "byte count")
    if any(not 0 < value <= 1 for value in probabilities):
        raise ValueError("probabilities must lie in (0, 1]")
    loss = finite_real(math.fsum(-math.log(p) for p in probabilities))
    return finite_real(loss/count/math.log(2))


def ratio_loss(selected_fraction, mean_score, target):
    fraction = finite_real(selected_fraction, "selected fraction")
    mean = finite_real(mean_score, "mean score")
    target = finite_real(target, "target ratio")
    if not (0 <= fraction <= 1 and 0 <= mean <= 1 and target > 1):
        raise ValueError("fractions in [0, 1] and target greater than 1")
    return finite_real(target*fraction*mean
                       + target/(target-1)*(1-fraction)*(1-mean))


def close_lists(first, second):
    assert len(first) == len(second)
    assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
               for a, b in zip(first, second))


def run():
    a = [0.5, 1, 0, 0.5]
    r = [2, 0, 7, 1]
    serial = serial_states(a, r)
    close_lists(serial, [2, 2, 7, 4.5])
    close_lists(scan_states(a, r), serial)
    close_lists(unrolled_states(a, r), serial)
    for initial in [0, -3, 8]:
        for count in range(1, 9):
            a = [0.5, 1, 0, -0.5]*2
            r = [2, 0, 7, 1]*2
            close_lists(scan_states(a[:count], r[:count], initial),
                        unrolled_states(a[:count], r[:count], initial))
    assert compose((2, 1), (3, 4)) == (6, 7)
    assert compose((3, 4), (2, 1)) == (6, 9)
    transition, gain = zoh_coefficients(2, math.log(2)/2, 1)
    assert math.isclose(transition, 0.5)
    assert math.isclose(gain, 0.25)
    assert zoh_coefficients(2, 0, 1) == (1, 0)
    assert zoh_coefficients(2, 1e-20, 1)[1] > 0
    assert math.isclose(zoh_coefficients(1e-300, 1e-300, 1)[1],
                        1e-300, rel_tol=1e-14, abs_tol=0)
    assert math.isclose(zoh_coefficients(1e-300, 1e-200, 1e100)[1],
                        1e-100, rel_tol=1e-14, abs_tol=0)
    assert math.isclose(zoh_coefficients(.5, 2, 1e308)[1],
                        -math.expm1(-1)*2e307*10, rel_tol=1e-14, abs_tol=0)
    scores = [1, 0.1, 0.75, 0.1]
    selected, expanded = causal_expand([2, 99, 6, -99], scores)
    assert selected == [0, 2]
    close_lists(expanded, [2, 2, 5, 5])
    for future in [-1e6, 0, 1000]:
        changed = causal_expand([2, 99, 6, future], scores)[1]
        close_lists(changed[:3], expanded[:3])
        selected_future = causal_expand(
            [2, 99, 6, future], [1, 0.1, 0.75, 0.9])[1]
        close_lists(selected_future[:3], expanded[:3])
    assert sum([0, 10])/2 != sum([0, 20])/2
    assert math.isclose(bits_per_byte([1/16], 4), 1)
    assert math.isclose(bits_per_byte([0.5]*4, 4), 1)
    assert len("é") == 1 and len("é".encode("utf-8")) == 2
    assert math.isclose(ratio_loss(0.5, 0.5, 2), 1)
    assert math.isclose(ratio_loss(0.6, 0.4, 2), 0.96)
    assert math.isclose(2**(-0.06), 0.9592641193252643)
    assert 4096*4097//2 == 8390656
    assert 4096*128*2 == 1048576
    assert 1024*128 + 128*127//2 == 139200
    assert 65*8 > 64*8
    invalid = [
        lambda: serial_states([], []),
        lambda: scan_states([1], [1, 2]),
        lambda: serial_states([True], [1]),
        lambda: scan_states([math.inf], [1]),
        lambda: scan_states([10**1000], [1]),
        lambda: compose((1e308, 0), (1e308, 0)),
        lambda: causal_expand([1], [0]),
        lambda: causal_expand([1, 2], [1, -0.1]),
        lambda: causal_expand([1], [1, 0]),
        lambda: bits_per_byte([0.5], 0),
        lambda: bits_per_byte([0.5], True),
        lambda: bits_per_byte([0], 1),
        lambda: bits_per_byte([1.1], 1),
        lambda: bits_per_byte([0.5], 10**1000),
        lambda: ratio_loss(0.5, 0.5, 1),
        lambda: ratio_loss(math.nan, 0.5, 2),
        lambda: zoh_coefficients(0, 1, 1),
        lambda: zoh_coefficients(1, -1, 1),
    ]
    for invalid_call in invalid:
        try:
            invalid_call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input was accepted")
    print("serial / scan / unrolled states:", serial)
    print("constant-input coefficients:", transition, gain)
    print("selected positions:", selected, "; expanded:", expanded)
    print("equal byte-normalized losses: 1.000000 bits/byte")
    print("off-diagonal ratio loss: 0.960000")
    print("UTF-8 bytes for one code point:", len("é".encode("utf-8")))
    print("finite recall requirement: 520 bits > 512 state bits")
    print("All recurrence, causality and 18 input checks passed.")


if __name__ == "__main__":
    run()
