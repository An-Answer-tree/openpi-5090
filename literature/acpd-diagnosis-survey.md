# ACPD Diagnosis Survey

Three literature-backed risks guide the next experiments:

1. A variance floor prevents constant collapse but does not enforce cue
   decorrelation or task relevance (VICReg).
2. A strong privileged teacher may still provide targets that a weak-view
   student cannot infer (Robust Asymmetric Learning).
3. Fixed teacher guidance can need dynamic weighting when its usefulness varies
   by state or training stage (Teacher-Guided Reinforcement Learning).

The cheapest discriminating order is therefore: measure teacher advantage,
isolate Cue and ACL, then test task success. A broad hyperparameter sweep before
these checks would confound method failure with target quality and optimization.
