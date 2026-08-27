# Track C C3-MS-CC — Current-Contact-Conditioned Prediction

C3-MS-CC freezes the accepted C2-R physical shared space and predicts only shared Contact tokens `u_c`. It compares two structured sources: Action plus current causal Contact context (`A+H`) and Vision plus Action plus current causal Contact context (`V+A+H`). The deterministic predictor uses eight learned Contact target queries over typed source tokens; Contact targets, native future Contact state, labels, pair identity, and private residuals are forbidden inputs.

Four preregistered validation-only trials were run: base and physics/temporal/covariance-preserving objectives for each source. The source-selection reducer requires A+H to pass every validation gate and lie within 0.01 of the best utility; otherwise it selects the best V+A+H candidate. T1/V+A+H was frozen before the locked benchmark was loaded.

On the 17,504-row locked benchmark, both source sets passed Contact and force retention, shared-latent prediction, H-context usage, retrieval, and no-collapse gates. A+H passed shared Contact physics; selected V+A+H did not. Vision produced no stable semantic improvement over A+H, so the scientific classification is `A_PLUS_H_SUFFICIENT_VISION_OPTIONAL`.

The final decision is `C3MSCC_ACTION_TEMPORAL_FAIL`. The external immutable Original UniT tokenizer required to regenerate frozen A-R latents from reversed raw Action sequences was unavailable. The implemented shared-token reversal surrogate is reported but deliberately does not satisfy the exact temporal-use gate. This fail-closed result keeps C4 at `NOT READY`; C4, C5, C6, and M3 were not started.

Run the tracked verification with:

```bash
python -m pytest -q tests/tactile_unit/test_c3mscc_contact_context.py
python scripts/tactile_unit/audit_c3mscc_contract.py
python scripts/tactile_unit/visualize_c3mscc_contact_prediction.py
```
