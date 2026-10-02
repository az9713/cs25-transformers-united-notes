"""Original finite fixtures for CS25 chapter 1; Python 3 standard library."""
import math


def finite_real(value, name):
    """Return a finite binary64 value or a predictable input-contract error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ': real int/float required; bool is excluded')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(name + ': value is not representable as a float')
    if not math.isfinite(result):
        raise ValueError(name + ': finite representable value required')
    return result


def matrix(data, name):
    if (not isinstance(data, (list, tuple)) or not data
            or not all(isinstance(row, (list, tuple)) for row in data)):
        raise ValueError(name + ': nonempty rows required')
    width = len(data[0])
    if not width or any(len(row) != width for row in data):
        raise ValueError(name + ': rectangular nonempty matrix required')
    for row in data:
        for value in row:
            finite_real(value, name)
    return width


def softmax(logits, allowed):
    if not isinstance(logits, (list, tuple)) or not logits:
        raise ValueError('finite logits required')
    logits = [finite_real(value, 'logit') for value in logits]
    if (not isinstance(allowed, (list, tuple)) or not allowed
            or any(type(j) is not int or not 0 <= j < len(logits)
                   for j in allowed)
            or len(set(allowed)) != len(allowed)):
        raise ValueError('nonempty unique permitted indices required')
    maximum = max(logits[j] for j in allowed)
    terms = [math.exp(logits[j] - maximum) for j in allowed]
    denominator = sum(terms)
    result = [0.0] * len(logits)
    for j, value in zip(allowed, terms):
        result[j] = value / denominator
    return result


def attention(queries, keys, values, causal=False):
    width = matrix(queries, 'queries')
    if matrix(keys, 'keys') != width:
        raise ValueError('query and key widths must agree')
    value_width = matrix(values, 'values')
    if len(values) != len(keys) or len(queries) != len(keys):
        raise ValueError('this self-attention fixture needs equal row counts')
    if type(causal) is not bool:
        raise ValueError('causal must be a boolean')
    queries, keys, values = [
        [[float(value) for value in row] for row in data]
        for data in (queries, keys, values)]
    weights, outputs = [], []
    for i, query in enumerate(queries):
        logits = [sum(a*b for a, b in zip(query, key)) / math.sqrt(width)
                  for key in keys]
        permitted = list(range(i+1 if causal else len(keys)))
        row = softmax(logits, permitted)
        weights.append(row)
        output = [sum(row[j]*values[j][k] for j in range(len(keys)))
                  for k in range(value_width)]
        outputs.append([finite_real(value, 'attention output')
                        for value in output])
    return weights, outputs


def mean_nll(probabilities):
    if not isinstance(probabilities, (list, tuple)) or not probabilities:
        raise ValueError('nonempty probabilities in (0, 1] required')
    probabilities = [finite_real(p, 'probability') for p in probabilities]
    if any(not 0 < p <= 1 for p in probabilities):
        raise ValueError('nonempty probabilities in (0, 1] required')
    return -sum(math.log(p) for p in probabilities) / len(probabilities)


def allocation(budget, minimum, pretrain_weight=4.0, index_weight=1.0):
    budget, minimum, pretrain_weight, index_weight = [
        finite_real(value, 'allocation input')
        for value in (budget, minimum, pretrain_weight, index_weight)]
    if not 0 < minimum < budget or min(pretrain_weight, index_weight) <= 0:
        raise ValueError('positive weights and 0 < minimum < budget required')
    root_pretrain = math.sqrt(pretrain_weight)
    root_index = math.sqrt(index_weight)
    pretrain = budget*(root_pretrain / (root_pretrain+root_index))
    pretrain = max(minimum, pretrain)
    index = budget-pretrain
    if not 0 < pretrain < budget or not index > 0:
        raise ValueError('positive allocation is not representable')
    return pretrain, index


def preference_loss(margin):
    margin = finite_real(margin, 'preference margin')
    return max(0.0, -margin) + math.log1p(math.exp(-abs(margin)))


def truth(claim, reference):
    if claim not in reference:
        return 'unknown'
    return 'true' if reference[claim] else 'false'


def action_allowed(statuses, lockout, confirmed):
    if (not statuses
            or any(s not in ('true', 'false', 'unknown') for s in statuses)):
        raise ValueError('nonempty known truth labels required')
    if type(lockout) is not bool or type(confirmed) is not bool:
        raise ValueError('lockout and confirmation must be boolean')
    return all(s == 'true' for s in statuses) and lockout and confirmed


def run():
    queries = [[1.0], [1.0], [1.0]]
    keys = [[0.0], [math.log(3)], [math.log(2)]]
    values = [[0.0], [6.0], [12.0]]
    weights, causal = attention(queries, keys, values, causal=True)
    _, unmasked = attention(queries, keys, values)
    assert all(math.isclose(sum(row), 1) for row in weights)
    assert all(row[j] == 0 for i, row in enumerate(weights)
               for j in range(i+1, len(row)))
    assert all(math.isclose(row[0], target)
               for row, target in zip(causal, [0, 4.5, 7]))
    assert math.isclose(unmasked[0][0], 7)
    changed = [[0.0], [6.0], [1200.0]]
    _, later = attention(queries, keys, changed, causal=True)
    assert causal[:2] == later[:2]
    order = [2, 0, 1]
    _, permuted = attention([queries[j] for j in order],
                            [keys[j] for j in order], [values[j] for j in order])
    assert all(math.isclose(permuted[i][0], unmasked[j][0])
               for i, j in enumerate(order))
    assert softmax([1000, 1000], [0, 1]) == [0.5, 0.5]
    print('causal outputs:', [round(row[0], 6) for row in causal])
    print('unmasked first output:', round(unmasked[0][0], 6))
    loss = mean_nll([0.5, 0.25, 0.125])
    assert math.isclose(loss, math.log(4))
    print('mean NLL:', round(loss, 6),
          'nats; perplexity:', round(math.exp(loss), 6))
    pretrain, index = allocation(12, 2)
    assert (pretrain, index) == (8, 4)
    risk = lambda d: 4/d + 1/(12-d)
    assert risk(8) < risk(6) and risk(8) < risk(10)
    assert allocation(12, 10) == (10, 2)
    print('toy allocation:', pretrain, 'pretrain;', index, 'index')
    assert math.isclose(preference_loss(math.log(3)), math.log(4/3))
    assert math.isclose(preference_loss(0), math.log(2))
    assert preference_loss(1000) == 0 and preference_loss(-1000) == 1000
    print('preference loss at log(3):', round(preference_loss(math.log(3)), 6))
    reference = {'latest_manual_used': True, 'bypass_interlock': False}
    statuses = [truth('latest_manual_used', reference),
                truth('part_fits', reference)]
    assert statuses == ['true', 'unknown']
    assert not action_allowed(statuses, True, True)
    assert not action_allowed(['true'], True, False)
    assert not action_allowed(['false'], True, True)
    assert action_allowed(['true'], True, True)
    print('evidence labels:', statuses, '; action permitted:', False)
    failures = [lambda: softmax([1, 2], []),
                lambda: softmax([1, 2], [0, 0]),
                lambda: attention([[1, 2]], [[1]], [[1]]),
                lambda: attention([[math.nan]], [[1]], [[1]]),
                lambda: attention([[10**1000]], [[1]], [[1]]),
                lambda: attention([[1e308]], [[1e308]], [[1]]),
                lambda: allocation(12, 2, 1e308, 1e-308),
                lambda: mean_nll([0]), lambda: allocation(12, 12),
                lambda: softmax([True], [0]),
                lambda: preference_loss(10**1000),
                lambda: action_allowed(['maybe'], True, True)]
    for operation in failures:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input was accepted')
    print('All chapter 1 numerical and input checks passed.')


if __name__ == '__main__':
    run()
