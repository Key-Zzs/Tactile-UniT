# S4.3-PI2N VAC diagnosis and bounded refinement

PI2N keeps VAC as the research question while treating the fixed-seed PI2M B_HVA result as a strong competing control. It does not assume that VAC must win and does not reinterpret the B2 result as a universal failure of contact representations.

The stage separates three questions: whether the frozen teacher carries task-domain contact information, whether Vision-side versus Contact-side readout of the existing shared space changes policy learning, and whether the authorized auxiliary readout or loss scale is mismatched. The policy budget is one matched no-H control (`B_VA27`), one Vision-side VAC candidate (`B_VAC_V`), and at most one diagnosis-routed refinement. No teacher weights, historical checkpoints, PI2B seed expansion, second task, or real robot are in scope.

The canonical time contract is the actual 50 Hz control timeline: `+27` raw/control rows is 0.54 s. The old BVA artifact remains a separate `+16` row / 0.32 s historical result. Every new target uses the exact 40,065 official rows, episode-local `+27` validity, and the same 2,700 masked tail rows without deleting behavior-cloning samples.

The frozen S4.2 B3 VAC bridge jointly optimized its Vision, Action, and Contact projectors. Its loss includes bidirectional Vision–Contact alignment and the optimizer contains every projector, so Contact influence on the Vision path is supported by training provenance. The clean VA-only and VAC teacher packages nevertheless differ in architecture, source data, and optimization budget. PI2N therefore labels their comparison `PACKAGE_LEVEL_TEACHER_COMPARISON_ONLY`; a policy difference cannot be attributed solely to adding Contact alignment.

The tracked preregistration is split across `configs/simulation/s4_3_pi2n_dag.json`, `s4_3_pi2n_diagnostics.json`, `s4_3_pi2n_candidate_protocol.json`, and `s4_3_pi2n_evaluation_protocol.json`. Numerical diagnostics and policy results belong in ignored, hash-bound artifacts. Planned work is not reported as completed evidence.
