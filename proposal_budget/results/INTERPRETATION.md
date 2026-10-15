# Interpretation boundary

This post-hoc evaluation treats each final scored detection as a proposed
sorting action under a fixed per-image action budget. It measures unique
non-crowd target coverage and how the limited proposal queue is divided among
valid, duplicate, and other-invalid detections.

It is not a physical robot experiment. The results do not measure grasp
success, realized material recovery, cycle time, collision avoidance,
graspability, or closed-loop sorting performance. The appropriate claim is
limited to the quality of detection-driven sorting proposals under the stated
offline matching rule.
