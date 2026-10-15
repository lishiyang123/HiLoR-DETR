# Protocol amendment v1.1

Status: **FROZEN BEFORE MEASUREMENT**. Claimed sensitivity results: **0**.

The asset inventory demonstrated that the existing 60-row UMR
grid (`bf062a4d15f33c5309f42c603afee428267d25e473dc5d85374f3dd67bd62622`) was generated before the disjoint-competitor feature
semantics were corrected. It is retained only as historical provenance and is
excluded from the final sensitivity table, figure, and claims.

Under v1.1, `pw=4` remains fixed by the already frozen final method. The UMR
sensitivity experiment evaluates only the frozen tau-by-beta grid using the
three SHA-verified corrected pw4 heads and the frozen detector-seed0 Val
capture. It does not refit UMR heads or retrain the detector.

All other v1.0 choices remain unchanged: detector seed0 only, LCCT delta
`0.01/0.02/0.04`, QCCR lambda `1e-4/1e-3/1e-2/1e-1` with original OOF-Val
rules, fixed `topk=300`, and complete Test prohibition.
