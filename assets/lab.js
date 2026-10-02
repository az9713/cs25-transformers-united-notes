/* Original finite attention fixture; no network or model inference. */
(() => {
  'use strict';
  function attention(row, futureValue, causal) {
    if (!Number.isInteger(row) || row < 0 || row > 2 || !Number.isFinite(futureValue) || typeof causal !== 'boolean') {
      throw new RangeError('Choose a query position, a finite value and a boolean mask.');
    }
    const logits = [0, Math.log(3), Math.log(2)];
    const allowed = logits.map((_, j) => !causal || j <= row);
    const maximum = Math.max(...logits.filter((_, j) => allowed[j]));
    const terms = logits.map((z, j) => allowed[j] ? Math.exp(z - maximum) : 0);
    const denominator = terms.reduce((a, b) => a + b, 0);
    const weights = terms.map(x => x / denominator);
    const values = [0, 6, futureValue];
    return {weights, output: weights.reduce((sum, a, j) => sum + a * values[j], 0)};
  }
  window.cs25Attention = attention;
  const position = document.getElementById('query-position');
  if (!position) return;
  const value = document.getElementById('future-value');
  const causal = document.getElementById('causal-mask');
  const output = document.getElementById('attention-output');
  function render() {
    if (!value.checkValidity() || !Number.isFinite(value.valueAsNumber)) {
      output.textContent = 'Enter a finite third value between -1000000 and 1000000.';
      return;
    }
    const result = attention(Number(position.value), value.valueAsNumber, causal.checked);
    output.textContent = `Weights: [${result.weights.map(x => x.toFixed(6)).join(', ')}]; output: ${result.output.toFixed(6)}.`;
  }
  [position, value, causal].forEach(control => control.addEventListener('input', render));
  render();
})();
