# Full Track C — C4 Missing-Modality Robustness and Uncertainty

C4 freezes the accepted A+H Contact predictor and adds two source-isolated
fallbacks for explicit missing-current-Contact conditions: Action-only and
offline Vision+Action. Availability is never inferred from tensor contents.
No Action produces an explicit abstention.

Selection used train and validation only. Five bounded trials were run: the
four preregistered A/VA base and physics/covariance candidates, plus one allowed
VA loss-weight variant prompted by the validation boundary. The selected VA
fallback was frozen before uncertainty training and before locked benchmark
access. A 19,097-parameter mode-aware estimator was then fit to per-sample
shared Contact error, calibrated with one common validation-only scale, and
frozen before the locked evaluation.

On the locked 17,504-row post-hoc benchmark, VA achieved Contact retention
0.507434 and force retention 0.696965. It passed shared-latent, retrieval,
teacher-side dynamic-physics, exact raw-Action temporal-use, invalid-H misuse,
and noncollapse gates. Vision improved Contact macro-F1 over A by 0.016572
(95% bootstrap CI 0.011536–0.021364), establishing a material missing-H gain.

Canonical fallback uncertainty achieved Spearman 0.777677, high-error AUROC
0.825688, and reduced shared error by 22.20% when the highest-uncertainty 20%
was removed. Its calibrated NLL beat constant variance, and its mean
uncertainty exceeded full-mode uncertainty with positive bootstrap evidence.

The frozen full path reproduced Contact retention 0.900433 and force retention
0.794947, with H use, exact Action temporal use, physics, and checkpoint
identity intact. Effective rank remains materially below the oracle, so the
rank warning is retained.

Decision: `C4_READY_VA_FALLBACK`. C5 is ready but was not started. The VA
fallback still consumes future-derived offline Vision and is not online-ready;
C5 remains mandatory. C6/M3 was not started and M3 is not established.
