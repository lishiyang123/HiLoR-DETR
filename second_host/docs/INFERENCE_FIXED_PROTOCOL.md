# Fixed CCVR fitting protocol

CCVR is fitted separately to each detector checkpoint. The `qccr` and `umr`
serialized names identify the competition mapping and candidate-value estimator
within CCVR; they do not specify extra independently proposed modules.

- Competition mapping: fixed regularization lambda=0.001; Val captures; five
  image-level folds for fitting diagnostics, then coefficients fitted on full Val.
- Candidate-value estimator: 25-dimensional host-adapted features plus final
  decoder hidden states; MLP; positive weight=4; head seed=0.
- Internal Train split: 938 fitting images and 104 early-stopping images;
  split seed=0. No Val or Test early stopping for the value head.
- Value training: at most 50 epochs, patience=8, AdamW lr=0.001, weight
  decay=1e-4, batch=16 images, minimum loss improvement=1e-6.
- Inference: tau=0, beta=2, Top-K=300; no Val grid or gate is used to select them.
- Complete captures retain all 300 final-layer query boxes and query-class logits.

Each fitted mapping/head is bound to its detector seed and checkpoint. The
stored binding verifies the model dimensions, feature schema, input capture
identities and fitted weights before inference. Whole-release source integrity
can be checked separately with the optional release verifier. Compatibility
status fields in serialized artifacts are explicitly marked as fixed-protocol
fields, not evidence of an observed Val acceptance gate.

CCVR changes candidate scores and ranking, not predicted box coordinates.
Test predictions are generated after these settings and fitted assets are fixed.
