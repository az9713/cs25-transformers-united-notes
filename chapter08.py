"""Finite multimodal architecture checks; no trained model or diffusion sampler.

Run with Python 3.10+ and the standard library. Dimensions and costs describe
the explicit toy contracts below, not any cited model's complete implementation.
"""
import math
from fractions import Fraction


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite real number')
    if not math.isfinite(value):
        raise ValueError(name + ' must be a finite real number')
    return float(value)


def patches(height, width, patch, frames=1):
    """Independent frames, spatial right/bottom padding, no special tokens."""
    height = positive_integer(height, 'height')
    width = positive_integer(width, 'width')
    patch = positive_integer(patch, 'patch')
    frames = positive_integer(frames, 'frames')
    return ((height + patch - 1) // patch) * ((width + patch - 1) // patch) * frames


def hybrid_mask(blocks):
    """None = text; each integer identifies ONE contiguous image block.

    Text attends causally. Image patches also attend within their image block.
    No packed independent examples, special tokens or duplicated clean/noisy
    copies are represented by this toy mask.
    """
    if not isinstance(blocks, (list, tuple)) or not blocks:
        raise ValueError('blocks must be a nonempty sequence')
    seen, previous = set(), None
    for block in blocks:
        if block is not None:
            if isinstance(block, bool) or not isinstance(block, int) or block < 0:
                raise ValueError('image block IDs must be nonnegative integers')
            if block != previous:
                if block in seen:
                    raise ValueError('each image block must be contiguous')
                seen.add(block)
        previous = block
    return [[j <= i or (block is not None and block == blocks[j])
             for j in range(len(blocks))] for i, block in enumerate(blocks)]


def softmax(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('softmax needs a nonempty sequence')
    values = [finite(x, 'score') for x in values]
    maximum = max(values)
    weights = [math.exp(x - maximum) for x in values]
    normalizer = math.fsum(weights)
    return [x / normalizer for x in weights]


def attend(queries, keys, values, mask):
    """One scalar head (d_k=d_v=1); arbitrary finite Q/K/V projections."""
    if not queries or not (len(queries) == len(keys) == len(values) == len(mask)):
        raise ValueError('Q/K/V/mask lengths must match and be nonzero')
    n = len(queries)
    queries = [finite(x, 'query') for x in queries]
    keys = [finite(x, 'key') for x in keys]
    values = [finite(x, 'value') for x in values]
    outputs, weights = [], []
    for i, row in enumerate(mask):
        if len(row) != n or any(type(x) is not bool for x in row) or not any(row):
            raise ValueError('mask rows need n booleans and a permitted key')
        indices = [j for j, permitted in enumerate(row) if permitted]
        local = softmax([queries[i] * keys[j] for j in indices])
        full = [0.0] * n
        for j, weight in zip(indices, local):
            full[j] = weight
        weights.append(full)
        outputs.append(math.fsum(weight * values[j] for j, weight in zip(indices, local)))
    return outputs, weights


def quantize(value, codebook):
    """Nearest scalar prototype; true ties select the lowest codebook index.

    Compare exact rational distances of the validated binary floats. Ordinary
    float subtraction can overflow or round distinct distances to a false tie.
    This finite teaching lookup prioritizes correctness over lookup throughput.
    """
    value = finite(value, 'value')
    if not isinstance(codebook, (list, tuple)) or not codebook:
        raise ValueError('codebook must be a nonempty sequence')
    codebook = [finite(x, 'prototype') for x in codebook]
    exact_value = Fraction(value)
    return min(range(len(codebook)),
               key=lambda i: abs(exact_value - Fraction(codebook[i])))


def binary_text_loss(feature, coefficient, target):
    """One TEXT target logit a*u; u may have come from an image encoder.

    Returns stable binary cross entropy and its derivative with respect to u.
    There is no image reconstruction loss here.
    """
    feature = finite(feature, 'feature')
    coefficient = finite(coefficient, 'coefficient')
    if type(target) is not int or target not in (0, 1):
        raise ValueError('target must be integer 0 or 1')
    logit = finite(feature * coefficient, 'logit')
    probability = (1 / (1 + math.exp(-logit)) if logit >= 0
                   else math.exp(logit) / (1 + math.exp(logit)))
    loss = max(logit, 0) - target * logit + math.log1p(math.exp(-abs(logit)))
    return loss, coefficient * (probability - target)


def gaussian_noise_predictor(noisy_value, alpha, clean_variance):
    """Analytic E[epsilon|x_tau] for independent zero-mean scalar Gaussians.

    epsilon has variance 1; clean x_0 has the supplied positive variance.
    x_tau = sqrt(alpha)*x_0 + sqrt(1-alpha)*epsilon.
    This is a denoising-regression oracle, NOT a reverse-process sampler.
    """
    noisy_value = finite(noisy_value, 'noisy_value')
    alpha = finite(alpha, 'alpha')
    clean_variance = finite(clean_variance, 'clean_variance')
    if not 0 < alpha < 1 or clean_variance <= 0:
        raise ValueError('need 0 < alpha < 1 and clean_variance > 0')
    variance = alpha * clean_variance + 1 - alpha
    coefficient = math.sqrt(1 - alpha) / variance
    residual_mse = alpha * clean_variance / variance
    return coefficient * noisy_value, residual_mse


def parameter_cost(width, hidden, modalities, tokens):
    """Square Q/K/V/O + NON-GATED two-matrix FFN; biases/norms omitted.

    All modalities use the same widths; exactly one branch per token.
    Counts matrix multiply-accumulate terms, not FLOPs or measured latency.
    """
    for value, name in [(width, 'width'), (hidden, 'hidden'),
                        (modalities, 'modalities'), (tokens, 'tokens')]:
        positive_integer(value, name)
    branch = 4 * width * width + 2 * width * hidden
    return branch * modalities, branch * tokens, 2 * tokens * tokens * width


def invalid_contracts():
    tests = [lambda: patches(0, 10, 2), lambda: patches(True, 10, 2),
             lambda: patches(10, 10, 2.5), lambda: patches(10, 10, 2, -1),
             lambda: hybrid_mask([]), lambda: hybrid_mask([0, None, 0]),
             lambda: hybrid_mask([True]), lambda: softmax([math.inf]),
             lambda: softmax([]), lambda: quantize(1, []),
             lambda: quantize(math.nan, [0]),
             lambda: binary_text_loss(1, 1, True),
             lambda: binary_text_loss(1e308, 1e308, 1),
             lambda: gaussian_noise_predictor(1, 0, 1),
             lambda: gaussian_noise_predictor(1, 1, 1),
             lambda: gaussian_noise_predictor(1, .5, 0),
             lambda: parameter_cost(8, 16, 0, 20),
             lambda: attend([0], [0], [], [[True]]),
             lambda: attend([0], [0], [1], [[False]])]
    for test in tests:
        try:
            test()
        except ValueError:
            continue
        raise AssertionError('invalid input accepted')
    return len(tests)


def main():
    image = patches(512, 512, 16)
    video = patches(512, 512, 16, 12)
    n_image, n_video = 512 + image, 512 + video
    print(f'patches: image={image}, video12={video}, padded17x17={patches(17,17,16)}')
    print(f'one-head BF16 scores: image={2*n_image*n_image/2**20:.3f} MiB, '
          f'video={2*n_video*n_video/2**20:.3f} MiB')
    blocks = [None, 0, 0, None, 1, 1, None]
    mask = hybrid_mask(blocks)
    assert mask[1][2] and not mask[1][3] and not mask[1][4]
    assert mask[4][5] and not mask[3][4]
    changed = [1, 2, 4, 8, 16, 32, 64]
    baseline, _ = attend([0]*7, [0]*7, changed, mask)
    changed[4:7] = [1000, 2000, 4000]
    altered, _ = attend([0]*7, [0]*7, changed, mask)
    assert baseline[:4] == altered[:4]
    print('mask: image-local future allowed; future text/image blocked; prefix invariant')
    indices = [quantize(x, [0, 1]) for x in [.1, .2]]
    assert indices == [0, 0]
    print(f'quantization: opposite task labels collide at code indices {indices}; Bayes error=0.500')
    assert quantize(1e308, [-1e308, -9e307]) == 1
    assert quantize(-1e308, [1, 0]) == 1
    assert quantize(.5, [0, 1]) == 0
    print('prototype lookup: overflow and rounded-distance false ties avoided; true ties use first index')
    loss, gradient = binary_text_loss(1, 1, 1)
    step = 1e-6
    difference = (binary_text_loss(1+step, 1, 1)[0]
                  - binary_text_loss(1-step, 1, 1)[0])/(2*step)
    assert abs(gradient - difference) < 1e-8
    print(f'text-only target: CE={loss:.6f}, image-feature gradient={gradient:.6f}')
    pure_mask = hybrid_mask([None, None])
    pure_before, _ = attend([0,0], [0,0], [1,3], pure_mask)
    pure_after, _ = attend([0,0], [0,0], [1,3], pure_mask)
    mixed_mask = hybrid_mask([None,0,0,None])
    before, weights = attend([0]*4, [0]*4, [1,2,2,3], mixed_mask)
    after, _ = attend([0]*4, [0]*4, [1,10,10,3], mixed_mask)
    assert pure_before == pure_after and before[-1] != after[-1]
    print(f'frozen text: pure={pure_before[-1]:.3f} unchanged; '
          f'mixed={before[-1]:.3f}->{after[-1]:.3f}; image attention mass={weights[-1][1]+weights[-1][2]:.3f}')
    prediction, mse = gaussian_noise_predictor(2, .25, 1)
    print(f'Gaussian denoising oracle: predicted noise={prediction:.6f}, irreducible MSE={mse:.6f}')
    stored, active, attention = parameter_cost(8,16,2,20)
    print(f'toy MoT: stored={stored} parameters, active projection/FFN={active} MACs, '
          f'global attention={attention} MACs')
    print(f'invalid inputs rejected: {invalid_contracts()}; all finite architecture checks passed')


if __name__ == '__main__':
    main()
