"""Original finite serving models. No GPU, model, network or framework required."""
import math


def finite(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('expected a finite number')
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError('number outside its nonnegative/positive domain')
    return value


def count(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError('expected an integer')
    return finite(value, positive=positive)


def token_metrics(arrival, timestamps):
    finite(arrival)
    if not timestamps:
        raise ValueError('a completed request needs at least one token')
    times = [finite(t) for t in timestamps]
    if times[0] < arrival or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('timestamps must follow arrival and be ordered')
    gaps = [b-a for a, b in zip(times, times[1:])]
    return dict(ttft=times[0]-arrival, ttl=times[-1]-arrival, gaps=gaps,
                tpot=sum(gaps)/len(gaps) if gaps else None)


def quantile(values, probability):
    """Empirical nearest-rank convention, including min at probability zero."""
    finite(probability)
    if probability > 1 or not values:
        raise ValueError('probability in [0,1] and nonempty samples required')
    ordered = sorted(finite(x) for x in values)
    return ordered[max(0, math.ceil(probability*len(ordered))-1)]


def fcfs(arrivals, services):
    """One work-conserving FCFS server, seconds, no preemption or batching."""
    if len(arrivals) != len(services):
        raise ValueError('one service time per arrival required')
    a = [finite(x) for x in arrivals]
    s = [finite(x, positive=True) for x in services]
    if any(y < x for x, y in zip(a, a[1:])):
        raise ValueError('arrivals must be ordered')
    available, rows = 0.0, []
    for arrival, service in zip(a, s):
        start = max(available, arrival)
        available = start + service
        rows.append((start-arrival, available-arrival, start, available))
    return rows


def mm1_mean(arrival_rate, service_rate):
    """Stationary M/M/1 response mean; deliberately not a GPU simulator."""
    finite(arrival_rate)
    finite(service_rate, positive=True)
    if arrival_rate >= service_rate:
        raise ValueError('no stationary response mean at load >= 1')
    return 1/(service_rate-arrival_rate)


def kv_bytes(layers, kv_heads, head_dim, bytes_per_element, lengths, block=1):
    for n in (layers, kv_heads, head_dim, block):
        count(n, positive=True)
    finite(bytes_per_element, positive=True)
    lengths = [count(n) for n in lengths]
    slots = sum(((n+block-1)//block)*block for n in lengths)
    return 2*layers*kv_heads*head_dim*bytes_per_element*slots


def roofline_seconds(flops, moved_bytes, peak_flops, bandwidth):
    for x in (flops, moved_bytes):
        finite(x)
    for x in (peak_flops, bandwidth):
        finite(x, positive=True)
    return max(flops/peak_flops, moved_bytes/bandwidth)


def categorical(values):
    if not values:
        raise ValueError('nonempty categorical distribution required')
    values = tuple(finite(x) for x in values)
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError('probabilities must sum to one')
    return values


def speculative_distribution(target, draft):
    """Enumerate accepted and residual mass, rather than Monte Carlo sampling."""
    p, q = categorical(target), categorical(draft)
    if len(p) != len(q):
        raise ValueError('target and draft need the same token alphabet')
    accepted = tuple(min(a, b) for a, b in zip(p, q))
    residual_mass = tuple(max(a-b, 0) for a, b in zip(p, q))
    rejection = math.fsum(residual_mass)
    residual = tuple(x/rejection for x in residual_mass) if rejection else None
    emitted = tuple(a+b for a, b in zip(accepted, residual_mass))
    assert all(math.isclose(a, b, abs_tol=1e-12) for a,b in zip(emitted,p))
    return accepted, rejection, residual, emitted


def block_speedup(acceptance, draft_tokens, relative_draft_cost):
    """IID constant acceptance; verification cost one target step; bonus token."""
    finite(acceptance)
    count(draft_tokens)
    finite(relative_draft_cost)
    if acceptance > 1:
        raise ValueError('acceptance in [0,1] required')
    advance = math.fsum(acceptance**i for i in range(draft_tokens+1))
    return advance/(1+draft_tokens*relative_draft_cost)


def main():
    trace = token_metrics(0, [0.4,0.42,0.44,0.9])
    print('Trace TTFT/TPOT/TTLT:', *(f'{trace[k]:.6f}' for k in ('ttft','tpot','ttl')))
    print('Trace p50/p95 gap:', quantile(trace['gaps'],.5), quantile(trace['gaps'],.95))
    assert token_metrics(1,[1.5])['tpot'] is None
    smooth = fcfs([0,1,2,3], [0.75]*4)
    burst = fcfs([0,0,0,0], [0.75]*4)
    print('FCFS smooth/burst response means:',
          *(f'{sum(r[1] for r in rows)/4:.6f}' for rows in (smooth,burst)))
    print('M/M/1 means at loads .5/.9/.99:',
          *(f'{mm1_mean(4*r,4):.6f}' for r in (.5,.9,.99)))
    exact=kv_bytes(32,8,128,2,[17,31])
    paged=kv_bytes(32,8,128,2,[17,31],block=16)
    print('KV exact/paged bytes:',exact,paged)
    print('KV fourfold query-head substitution error:',kv_bytes(32,32,128,2,[17,31])/exact)
    print('Roofline decode/prefill seconds:',
          *(f'{roofline_seconds(f,d,1e14,1e12):.6f}' for f,d in ((2e9,8e9),(8e12,8e9))))
    print('Weight-only halving, original weight fraction .6:',f'{1/(.4+.6/2):.6f}')
    p,q=(.6,.3,.1),(.2,.3,.5)
    accepted,rejected,residual,emitted=speculative_distribution(p,q)
    print('Speculative accepted/rejection/residual:',accepted,f'{rejected:.6f}',residual)
    print('Speculative corrected output:',tuple(round(x,6) for x in emitted))
    assert speculative_distribution(p,p)[2] is None
    assert speculative_distribution((1,0),(0,1))[3] == (1,0)
    for a in (0,.5,.8,1):
        print(f'Block k=4, draft cost .1, acceptance {a:g}: {block_speedup(a,4,.1):.6f}')
    assert fcfs([],[]) == []
    assert kv_bytes(1,1,1,2,[]) == 0
    assert roofline_seconds(0,0,1,1) == 0
    assert block_speedup(1,0,0) == 1
    invalid = [lambda: token_metrics(0,[]),lambda: token_metrics(1,[.5]),
        lambda: token_metrics(0,[1,.5]),lambda: quantile([], .5),
        lambda: quantile([1],1.1),lambda: fcfs([1,0],[1,1]),
        lambda: fcfs([0],[0]),lambda: fcfs([0],[]),
        lambda: mm1_mean(4,4),lambda: mm1_mean(-1,4),
        lambda: kv_bytes(1,1,1,2,[1],0),lambda: kv_bytes(1,1,1,2,[1.5]),
        lambda: roofline_seconds(1,1,0,1),lambda: categorical([.1,.2]),
        lambda: categorical([float('nan'),1]),lambda: speculative_distribution([1],[0,1]),
        lambda: block_speedup(1.1,4,.1),lambda: block_speedup(.5,True,.1)]
    for call in invalid:
        try: call()
        except ValueError: pass
        else: raise AssertionError('invalid input accepted')
    print('Boundary cases and',len(invalid),'invalid-input cases passed.')


if __name__ == '__main__':
    main()
