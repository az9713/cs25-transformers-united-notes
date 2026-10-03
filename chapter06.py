"""Original finite fixtures for context, deduction and accessible information.

No model, retrieval service, training procedure or biological system is simulated.
Run with Python 3; standard library only.
"""
import math


def probability(value):
    """Accept finite binary64 real inputs in [0, 1], excluding bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('expected a real probability')
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError('probability is outside float range') from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('probability must be finite and in [0, 1]')
    return result


def bounded_count(value, minimum=0):
    if type(value) is not int or not minimum <= value <= 10**9:
        raise ValueError('expected an integer count within the stated bound')
    return value


def fact_set(rows):
    if not isinstance(rows, (list, tuple, set, frozenset)):
        raise ValueError('facts must be a finite explicit collection')
    facts = set()
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError('each fact must be a subject/relation/object triple')
        if any(type(x) is not str or not x.strip() for x in row):
            raise ValueError('fact fields must be nonempty strings')
        facts.add(tuple(row))
    return facts


def inverse_lookup(rows, relation, target):
    """Read explicit facts backwards; no claim about a neural model's recall."""
    facts = fact_set(rows)
    if any(type(x) is not str or not x.strip() for x in (relation, target)):
        raise ValueError('query fields must be nonempty strings')
    return sorted({a for a, r, b in facts if r == relation and b == target})


def deduce_is_a(rows, rounds=1):
    """Apply stated is-a transitivity for a bounded number of synchronous rounds."""
    facts = fact_set(rows)
    if type(rounds) is not int or not 0 <= rounds <= 20:
        raise ValueError('rounds must be an integer from 0 to 20')
    # ponytail: O(m^2) pair scans for small teaching graphs; use adjacency indexes
    # if this is repurposed for large graphs, with explicit work/storage budgets.
    for _ in range(rounds):
        additions = {(a, 'is_a', c) for a, r, b in facts for x, s, c in facts
                     if r == s == 'is_a' and b == x}
        if additions <= facts:
            break
        facts |= additions
    return sorted(facts)


def first_coordinate_nll(rows):
    """Empirical optimum log loss for all Bernoulli maps of the first bit.

    Each row is (three binary features, binary label). Natural logarithms.
    The family can ignore all features by using the same probability for both
    first-bit values. This computes its optimum, not a fitted neural model.
    """
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError('expected nonempty explicit examples')
    groups = {}
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError('expected a feature/label pair')
        features, label = row
        if not isinstance(features, (tuple, list)) or len(features) != 3:
            raise ValueError('expected three binary features')
        if any(type(x) is not int or x not in (0, 1)
               for x in (*features, label)):
            raise ValueError('features and label must be integer bits')
        counts = groups.setdefault(features[0], [0, 0])
        counts[label] += 1
    total = len(rows)
    loss = 0.0
    for counts in groups.values():
        size = sum(counts)
        for count in counts:
            if count:
                loss -= count / total * math.log(count / size)
    return loss


def evidence_accuracy(recall, with_evidence, without_evidence):
    """Total probability under three specified conditional rates."""
    r, a, b = map(probability, (recall, with_evidence, without_evidence))
    return r * a + (1 - r) * b


def enumeration_success(population, attempts, distinct=True):
    """Uniform candidates; distinct draws or independent draws with replacement."""
    n, k = bounded_count(population, 1), bounded_count(attempts)
    if type(distinct) is not bool:
        raise ValueError('distinct must be bool')
    if distinct:
        if k > n:
            raise ValueError('distinct attempts cannot exceed population')
        return k / n
    if n == 1:
        return float(k > 0)
    return -math.expm1(k * math.log1p(-1 / n))


def demo():
    facts = [('pump', 'requires', 'lockout'),
             ('pump', 'is_a', 'powered_asset'),
             ('powered_asset', 'is_a', 'maintained_asset')]
    assert inverse_lookup(facts, 'requires', 'lockout') == ['pump']
    assert inverse_lookup(facts, 'requires', 'pump') == []
    assert ('pump', 'is_a', 'maintained_asset') in deduce_is_a(facts)
    assert ('lockout', 'requires', 'pump') not in deduce_is_a(facts)
    assert deduce_is_a([], 20) == []
    assert deduce_is_a(facts, 0) == sorted(fact_set(facts))
    original = [((u, v, 0), u ^ v) for u in (0, 1) for v in (0, 1)]
    exposed = [((u ^ v, u, v), u ^ v) for u in (0, 1) for v in (0, 1)]
    initial, transformed = map(first_coordinate_nll, (original, exposed))
    assert math.isclose(initial, math.log(2)) and transformed == 0
    assert evidence_accuracy(0, 0.9, 0.2) == 0.2
    assert evidence_accuracy(1, 0.9, 0.2) == 0.9
    assert math.isclose(evidence_accuracy(.8, .9, .2), .76)
    assert enumeration_success(10, 3) == .3
    assert math.isclose(enumeration_success(10, 3, False), .271)
    assert enumeration_success(1, 0, False) == 0
    assert enumeration_success(1, 5, False) == 1
    bad_calls = [
        lambda: probability(True), lambda: probability(float('nan')),
        lambda: probability(float('inf')), lambda: probability(10**1000),
        lambda: evidence_accuracy(-.1, .9, .2),
        lambda: inverse_lookup([('x', 'r')], 'r', 'x'),
        lambda: inverse_lookup([('x', 'r', '')], 'r', 'x'),
        lambda: deduce_is_a(facts, True), lambda: deduce_is_a(facts, 21),
        lambda: first_coordinate_nll([]),
        lambda: first_coordinate_nll([((0, True, 0), 1)]),
        lambda: enumeration_success(0, 1),
        lambda: enumeration_success(10, 11),
        lambda: enumeration_success(True, 1),
        lambda: enumeration_success(10**1000, 1),
        lambda: enumeration_success(10, 1, 'yes'),
    ]
    for call in bad_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input was accepted')
    print('inverse explicit lookup:', inverse_lookup(facts, 'requires', 'lockout'))
    print('one-round deduction:', ('pump', 'is_a', 'maintained_asset') in deduce_is_a(facts))
    print(f'restricted NLL: {initial:.6f} -> {transformed:.6f} nats')
    print(f'conditional evidence accuracy: {evidence_accuracy(.8, .9, .2):.2f}')
    print(f'candidate coverage: {enumeration_success(10, 3):.3f} distinct; '
          f'{enumeration_success(10, 3, False):.3f} replacement')
    print('All chapter 6 finite-fixture and input checks passed.')


if __name__ == '__main__':
    demo()
