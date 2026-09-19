# Selected-parameter freezing

Freezer is a native Lightning callback that temporarily freezes selected parameters and restores their original `requires_grad` flags. Ordinary modules use parameter names from the module; LighterModule uses names relative to `network`.

## Select parameters

This callback-section excerpt assumes your network has the named parameters:

```yaml
trainer:
  callbacks:
    - _target_: lighter.callbacks.Freezer
      name_starts_with: encoder.backbone.
      except_names: encoder.backbone.special_adapter.weight
      until_epoch: 5
```

`names` matches complete parameter names; `name_starts_with` matches prefixes. Exceptions exclude parameters from this callback's ownership, including shared/tied aliases. They do not make those parameters trainable. Empty selections and overlapping ownership by multiple Freezers fail; disjoint selections can coexist.

## Release and resume

Omit both limits for indefinite freezing. Otherwise use exactly one nonnegative `until_epoch` or `until_step`. Steps count native optimizer updates, not batches. On release, originally frozen parameters stay frozen; unrelated flags remain untouched. Selected stale gradients are cleared so an optimizer cannot apply momentum or weight decay to an old gradient.

Callback checkpoint state preserves original flags and reapplies the schedule using restored progress. Resume with the same parameter names and callback policy. Freezer does not change optimizer groups: parameters that will reactivate must already be included. Use native `BaseFinetuning` when your policy intentionally inserts groups or changes their LR.

## Distributed constraints

Freezing occurs after strategy wrapping. DDP requires `find_unused_parameters=True` and `static_graph=False`; Lightning's `ddp_find_unused_parameters_true` is the ordinary configuration. Parameters that may reactivate must be trainable when DDP wraps the model.

Other strategies need their own native policy. Freezer does not silently rewrite Trainer strategies. See [execution limits](compatibility.md#exact-validation-profiles) and the [API reference](../reference/callbacks/freezer.md).

## Migration

Earlier Freezer versions forced unrelated/excepted parameters trainable. They now leave them untouched. Old checkpoints without callback state cannot restore flags that were never recorded; the new module's initial flags establish the baseline. Changing the callback policy creates a new policy, not an exact continuation.
