# Selected-parameter freezing

`Freezer` is a native Lightning callback. It temporarily freezes selected parameters, then restores their original `requires_grad` flags. Ordinary modules use parameter names from the module; `LighterModule` uses names relative to `network`.

```yaml
trainer:
  _target_: pytorch_lightning.Trainer
  callbacks:
    - _target_: lighter.callbacks.Freezer
      name_starts_with: encoder.backbone.
      except_names: encoder.backbone.special_adapter.weight
      until_epoch: 5
```

`names` matches complete parameter names (such as `encoder.weight`). `name_starts_with` matches prefixes. Exceptions exclude parameters from ownership, including shared/tied aliases; they do not make parameters trainable. Empty selections and overlapping ownership by two Freezer callbacks fail clearly. Disjoint callbacks can coexist.

Omit both limits for indefinite freezing. Otherwise use exactly one nonnegative `until_epoch` or `until_step`; the latter counts native optimizer steps, not batches. On release, a parameter that was originally frozen stays frozen. Other parameters keep their existing flags. Freezing clears stale selected gradients so an optimizer cannot apply momentum or weight decay to an old gradient.

The callback saves original flags through Lightning's callback checkpoint state and reapplies the schedule using restored progress. Resume with the same model parameter names and callback policy. The callback deliberately does not change optimizer groups: every parameter that will reactivate must already be in an optimizer. Native `BaseFinetuning` is available when a project intentionally needs parameter-group insertion or its own learning-rate policy.

Freezing occurs after strategy wrapping. DDP requires `find_unused_parameters=True` and `static_graph=False`; use Lightning's `ddp_find_unused_parameters_true` strategy for the ordinary configuration. Parameters that can reactivate must be trainable when DDP wraps the model. Other distributed strategies require their own native policy and are not automatically supported by this callback. These constraints protect gradient synchronization; the callback does not silently rewrite a Trainer strategy.

Migration: earlier Freezer versions forced all unrelated and excepted parameters trainable. That behavior could unfreeze a teacher or a separately managed submodule. They now remain untouched. Old checkpoints without Freezer callback state cannot recover flags that were never recorded; in that case the new module's initial flags establish the baseline. A checkpoint with a changed callback policy is a new policy, not an exact continuation.
