# Parameter sensitivity protocol v1.2

Status: **complete**. The detector-selection and sensitivity seed remains seed0.

The training-side LCCT grid is `0.005 / 0.01 / 0.02 / 0.04 / 0.08` with
Train used for fitting and native Val AP used for checkpoint selection and
reporting. The exact final seed0 run is reused at the frozen center
`delta=0.02`; four surrounding values are independent formal runs.

The inference-side rules remain those frozen in v1.1: QCCR uses its four
image-level five-fold OOF-Val candidates, and UMR fixes corrected-semantics
`pw=4` while evaluating only the `tau x beta` grid with head seeds 0/1/2.
The legacy pre-correction pw grid remains excluded. Test is prohibited for all
training, checkpoint selection, parameter selection, and sensitivity analysis.
