"""Original finite distributed-training simulator. Standard library only.

No GPUs, networks, files, training frameworks, or external effects are used.
Rank lists model physical copies; collectives use SUM, never implicit averaging.
"""

from fractions import Fraction
import math


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def number(value):
    if type(value) not in (int, float):
        raise ValueError("numeric values must be real int or float, excluding bool")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("numeric value is not representable") from exc
    if not math.isfinite(value):
        raise ValueError("numeric value must be finite")
    return value


def finite_sum(values):
    try:
        return number(math.fsum(values))
    except OverflowError as exc:
        raise ValueError("computed sum is not representable") from exc


def matrix(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("matrix must be a nonempty row sequence")
    if any(not isinstance(row, (list, tuple)) or not row for row in values):
        raise ValueError("matrix rows must be nonempty sequences")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("matrix must be rectangular")
    return [[number(x) for x in row] for row in values]


def transpose(values):
    return [list(row) for row in zip(*matrix(values))]


def matmul(left, right):
    left, right = matrix(left), matrix(right)
    if len(left[0]) != len(right):
        raise ValueError("matrix inner dimensions must match")
    cols = transpose(right)
    return [[finite_sum(number(a * b) for a, b in zip(row, col))
             for col in cols] for row in left]


def all_gather(shards):
    shards = matrix(shards)
    full = [x for shard in shards for x in shard]
    return [full[:] for _ in shards]


def reduce_scatter(copies):
    copies = matrix(copies)
    ranks, width = len(copies), len(copies[0])
    if width % ranks:
        raise ValueError("full width must be divisible by rank count")
    summed = [finite_sum(row[i] for row in copies) for i in range(width)]
    shard_width = width // ranks
    return [summed[r * shard_width:(r + 1) * shard_width] for r in range(ranks)]


def inner(left, right):
    left, right = matrix(left), matrix(right)
    if (len(left), len(left[0])) != (len(right), len(right[0])):
        raise ValueError("inner product shapes must match")
    return finite_sum(number(a * b) for x, y in zip(left, right)
                      for a, b in zip(x, y))


def global_mean(local_means, counts):
    if not isinstance(local_means, (list, tuple)) or not local_means:
        raise ValueError("local means must be a nonempty sequence")
    if not isinstance(counts, (list, tuple)) or len(counts) != len(local_means):
        raise ValueError("one positive count is required per local mean")
    counts = [positive_int(c, "count") for c in counts]
    total = sum(counts)
    return finite_sum(number(g) * (c / total) for g, c in zip(local_means, counts))


def column_parallel(left, weight, upstream, ranks):
    left, weight, upstream = matrix(left), matrix(weight), matrix(upstream)
    ranks = positive_int(ranks, "ranks")
    if len(left[0]) != len(weight):
        raise ValueError("input and weight dimensions must match")
    width = len(weight[0])
    if width % ranks or (len(upstream), len(upstream[0])) != (len(left), width):
        raise ValueError("output width must divide ranks and match upstream")
    step = width // ranks
    outputs, dx_parts, dw_parts = [], [], []
    for r in range(ranks):
        w = [row[r * step:(r + 1) * step] for row in weight]
        g = [row[r * step:(r + 1) * step] for row in upstream]
        outputs.append(matmul(left, w))
        dx_parts.append(matmul(g, transpose(w)))
        dw_parts.append(matmul(transpose(left), g))
    output = [[x for shard in outputs for x in shard[i]] for i in range(len(left))]
    dx = [[finite_sum(part[i][j] for part in dx_parts)
           for j in range(len(left[0]))] for i in range(len(left))]
    dw = [[x for shard in dw_parts for x in shard[i]] for i in range(len(weight))]
    return output, dx, dw


def block_attention(blocks):
    """One query, scalar values, supplied finite scores; each block is nonempty."""
    if not isinstance(blocks, (list, tuple)) or not blocks:
        raise ValueError("attention needs at least one block")
    running_max, denominator, numerator = None, 0.0, 0.0
    for block in blocks:
        if not isinstance(block, (list, tuple)) or not block:
            raise ValueError("each block needs a permitted key")
        pairs = []
        for pair in block:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("each attention pair is (score, scalar value)")
            pairs.append((number(pair[0]), number(pair[1])))
        local_max = max(score for score, _ in pairs)
        new_max = local_max if running_max is None else max(running_max, local_max)
        # Negative score differences may round to -inf; exp(-inf)=0 is intentional.
        old_scale = 0.0 if running_max is None else math.exp(running_max - new_max)
        weights = [math.exp(score - new_max) for score, _ in pairs]
        denominator = finite_sum([denominator * old_scale, finite_sum(weights)])
        numerator = finite_sum([number(numerator * old_scale),
                                finite_sum(number(w * v) for w, (_, v) in zip(weights, pairs))])
        running_max = new_max
    return number(numerator / denominator)


def state_bytes(parameters, ranks, stage):
    parameters, ranks = positive_int(parameters, "parameters"), positive_int(ranks, "ranks")
    if type(stage) is not int or stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0, 1, 2, or 3")
    # Ideal equal shards; Fraction exposes non-divisible padding as excluded overhead.
    coefficients = (Fraction(16), 4 + Fraction(12, ranks),
                    2 + Fraction(14, ranks), Fraction(16, ranks))
    return parameters * coefficients[stage]


def afab(stages, microbatches):
    """Balanced unit-duration forward/backward tasks with a global phase barrier."""
    stages = positive_int(stages, "stages")
    microbatches = positive_int(microbatches, "microbatches")
    if stages * microbatches > 10000:
        raise ValueError("finite schedule fixture permits at most 10000 stage-batch pairs")
    forward = [[0] * microbatches for _ in range(stages)]
    backward = [[0] * microbatches for _ in range(stages)]
    events = []
    for s in range(stages):
        for m in range(microbatches):
            start = max(forward[s][m - 1] if m else 0,
                        forward[s - 1][m] if s else 0)
            forward[s][m] = start + 1
            events.append((s, m, "F", start, start + 1))
    barrier = forward[-1][-1]
    for s in reversed(range(stages)):
        for m in range(microbatches):
            start = max(barrier, backward[s][m - 1] if m else barrier,
                        backward[s + 1][m] if s + 1 < stages else barrier)
            backward[s][m] = start + 1
            events.append((s, m, "B", start, start + 1))
    return backward[0][-1], events


def expert_load(routes, experts):
    experts = positive_int(experts, "experts")
    if not isinstance(routes, (list, tuple)) or not routes:
        raise ValueError("routes must contain tokens")
    loads, topk = [0] * experts, None
    for route in routes:
        if not isinstance(route, (list, tuple)) or not route:
            raise ValueError("each token requires expert IDs")
        if any(type(e) is not int or e < 0 or e >= experts for e in route):
            raise ValueError("expert ID must be an integer in range")
        if len(set(route)) != len(route):
            raise ValueError("a token cannot select an expert twice")
        if topk is None:
            topk = len(route)
        if len(route) != topk:
            raise ValueError("fixture requires constant top-k")
        for e in route:
            loads[e] += 1
    return loads


def checks():
    shards, copies = [[1, 2], [3, 4]], [[5, 6, 7, 8], [1, 2, 3, 4]]
    lhs, rhs = inner(all_gather(shards), copies), inner(shards, reduce_scatter(copies))
    assert lhs == rhs == 100.0
    assert reduce_scatter(all_gather(shards)) == [[2, 4], [6, 8]]
    assert all_gather(reduce_scatter(copies)) == [[6, 8, 10, 12]] * 2
    print(f"collective adjoint inner products: {lhs:.1f} = {rhs:.1f}")
    print("AG then RS: [[2.0, 4.0], [6.0, 8.0]]; not identity")
    weighted = global_mean([2, 6], [1, 3])
    assert weighted == 5.0 and global_mean([2, 6], [1, 1]) == 4.0
    print(f"global mean: weighted={weighted:.1f}, mean-of-means=4.0")
    x, w, g = [[1, 2], [3, 4]], [[5, 6], [7, 8]], [[1, 1], [1, 1]]
    out, dx, dw = column_parallel(x, w, g, 2)
    assert out == matmul(x, w) == [[19, 22], [43, 50]]
    assert dx == matmul(g, transpose(w)) == [[11, 15], [11, 15]]
    assert dw == matmul(transpose(x), g) == [[4, 4], [6, 6]]
    eps = 1e-5
    for i in range(2):
        for j in range(2):
            plus, minus = [r[:] for r in x], [r[:] for r in x]
            plus[i][j] += eps
            minus[i][j] -= eps
            fd = (finite_sum(sum(r) for r in matmul(plus, w)) -
                  finite_sum(sum(r) for r in matmul(minus, w))) / (2 * eps)
            assert math.isclose(fd, dx[i][j], rel_tol=1e-8, abs_tol=1e-8)
            w_plus, w_minus = [r[:] for r in w], [r[:] for r in w]
            w_plus[i][j] += eps
            w_minus[i][j] -= eps
            fd_w = (finite_sum(sum(r) for r in matmul(x, w_plus)) -
                    finite_sum(sum(r) for r in matmul(x, w_minus))) / (2 * eps)
            assert math.isclose(fd_w, dw[i][j], rel_tol=1e-8, abs_tol=1e-8)
    print(f"column TP output={out}; dX={dx}; dW={dw}")
    # Normalized token coordinates and upstreams are supplied, not an LN kernel.
    norm, upstream = [[1, -1], [-1, 1]], [[2, 3], [5, 7]]
    local_gamma = [[a * b for a, b in zip(n, u)] for n, u in zip(norm, upstream)]
    gamma = [finite_sum(r[j] for r in local_gamma) for j in range(2)]
    assert gamma == [-3, 4] and local_gamma[0] != local_gamma[1]
    print(f"sequence-local gamma gradients={local_gamma}; sum={gamma}")
    blocks = [[(1000, 2), (1001, 4)], [(999, -1)]]
    attention = block_attention(blocks)
    flat = block_attention([[p for b in blocks for p in b]])
    assert math.isclose(attention, flat, rel_tol=1e-15)
    oracle = (2 * math.exp(-1) + 4 - math.exp(-2)) / (math.exp(-1) + 1 + math.exp(-2))
    assert math.isclose(attention, oracle, rel_tol=1e-15)
    assert math.isclose(attention, block_attention(list(reversed(blocks))), rel_tol=1e-15)
    assert block_attention([[(2, 7)]]) == 7
    print(f"online attention={attention:.9f}; flat/reversed agree")
    gb = [float(state_bytes(7_000_000_000, 8, s) / 10**9) for s in range(4)]
    assert gb == [112, 38.5, 26.25, 14]
    assert all(state_bytes(17, 1, s) == 272 for s in range(4))
    print("7B model state GB at R=8: " + ", ".join(f"{v:.2f}" for v in gb))
    for p, m in [(1, 3), (4, 1), (4, 8), (3, 7)]:
        duration, events = afab(p, m)
        assert duration == 2 * (p + m - 1)
        for stage in range(p):
            local = sorted((a, b) for s, _, _, a, b in events if s == stage)
            assert all(a[1] <= b[0] for a, b in zip(local, local[1:]))
        for stage, batch, kind, start, _ in events:
            if (kind == "F" and stage > 0) or (kind == "B" and stage + 1 < p):
                producer = stage - 1 if kind == "F" else stage + 1
                producer_end = next(b for s, m0, k, _, b in events
                                    if (s, m0, k) == (producer, batch, kind))
                assert start >= producer_end
            if kind == "B":
                f_end = next(b for s, m0, k, _, b in events
                             if (s, m0, k) == (stage, batch, "F"))
                assert start >= f_end
    duration, _ = afab(4, 8)
    print(f"AFAB p=4 m=8: makespan={duration}, utilization={8 / 11:.6f}")
    loads = expert_load([[0, 1], [0, 1], [0, 2], [0, 3]], 4)
    assert loads == [4, 2, 1, 1] and sum(loads) == 8
    print(f"expert loads={loads}; max/mean={max(loads) / (sum(loads) / 4):.1f}")
    invalid = [lambda: positive_int(True, "ranks"), lambda: number(float("nan")),
               lambda: number(10**1000), lambda: matrix([]), lambda: matrix([[1], [2, 3]]),
               lambda: matrix([[True]]), lambda: matmul([[1]], [[1, 2], [3, 4]]),
               lambda: matmul([[1e308]], [[1e308]]),
               lambda: finite_sum([1e308, 1e308]),
               lambda: reduce_scatter([[1, 2, 3], [4, 5, 6]]),
               lambda: inner([[1]], [[1, 2]]), lambda: global_mean([2], [0]),
               lambda: global_mean([2, 3], [1]), lambda: column_parallel(x, w, g, 3),
               lambda: block_attention([]), lambda: block_attention([[]]),
               lambda: block_attention([[(1, float("inf"))]]),
               lambda: block_attention([[(1, 2, 3)]]), lambda: state_bytes(7, 0, 3),
               lambda: state_bytes(7, 2, True), lambda: afab(0, 1), lambda: afab(101, 100),
               lambda: expert_load([[0, 0]], 2), lambda: expert_load([[2]], 2),
               lambda: expert_load([[0], [0, 1]], 2)]
    for call in invalid:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input was accepted")
    print(f"invalid-input contracts: {len(invalid)} rejected")


if __name__ == "__main__":
    checks()
