# Privileged Visual Policy Distillation

| Work | Mechanism | Implication for ACPD |
|---|---|---|
| [Learning by Cheating](https://arxiv.org/abs/1912.12294) | A privileged state teacher supplies action supervision on states visited by the visual student. | Teacher strength is useful when supervision covers the student's state distribution; fixed offline feature matching does not provide this benefit. |
| [Multi-camera to Single-camera Visual Policy Distillation](https://arxiv.org/abs/2303.07026) | Multi-view teacher features supervise a single-view student, with camera randomization and three-seed evaluation. | Cross-view transfer is possible, but the target and augmentation must make privileged information recoverable from the student image. |
| [RMA](https://arxiv.org/abs/2107.04034) | A history-conditioned adaptation module predicts a privileged latent used by the policy. | When one frame is partially observable, history is a principled input for privilege recovery. |
| [Motion Planner Augmented Policy Distillation](https://proceedings.mlr.press/v164/liu22b.html) | Visual behavioral cloning first transfers the privileged policy, then task interaction refines it. | Output-level policy supervision and task optimization are more direct than unconstrained hidden-state matching. |
| [Theia](https://proceedings.mlr.press/v270/shang25a.html) | Distills diverse teacher visual representations before downstream policy learning. | Representation distillation should be evaluated for retained, noncollapsed visual information before policy training. |

## Working Rule

The next method must transfer a target that is both task-relevant and
recoverable at inference. H7 tests this for exact teacher attention
contributions. If recoverability fails mainly on teacher-advantage examples,
the next experiment should add backview history instead of increasing a
feature-loss weight. If recoverability passes, use one selected layer in a
matched 5K policy experiment before any broad hyperparameter search.

