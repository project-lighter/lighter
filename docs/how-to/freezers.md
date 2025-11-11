# Freezers: Smart Layer Management for Transfer Learning

Freezing layers is a powerful technique that can accelerate training, prevent catastrophic forgetting, and improve model performance. Lighter's `Freezer` callback gives you fine-grained control over which layers train and when.

## Quick Start 🚀

```yaml title="config.yaml"
# Freeze encoder for first 10 epochs, then unfreeze
trainer:
  callbacks:
    - _target_: lighter.callbacks.Freezer
      name_starts_with: ["model.encoder"]  # What to freeze
      until_epoch: 10                      # When to unfreeze
```

## Why Freeze Layers? 🤔

| Scenario | Strategy | Benefit |
|----------|----------|---------|
| **Transfer Learning** | Freeze pretrained layers initially | Preserve learned features |
| **Limited Data** | Freeze most layers | Prevent overfitting |
| **Fine-tuning** | Gradual unfreezing | Stable adaptation |
| **Multi-stage Training** | Stage-wise freezing | Focused learning |

**Freezing Strategies**:

`Freezer` callback offers flexible layer freezing strategies:

1.  **Freeze by Name Prefix (`name_starts_with`)**:

    *   Freeze parameters with names starting with prefix/prefixes in `name_starts_with` arg.
    *   Useful for freezing modules or layer groups.
    *   **Example**:

        ```yaml title="config.yaml"
        trainer:
          callbacks:
            - _target_: lighter.callbacks.Freezer
              name_starts_with: ["model.encoder", "model.embedding"] # Freeze encoder/embedding layers
              until_epoch: 5
        ```

        Config freezes parameters starting with `"model.encoder"` or `"model.embedding"` until epoch 5.

2.  **Freeze by Exact Name (`names`)**:

    *   Freeze specific parameters by name using `names` arg.
    *   For fine-grained control over individual layers/parameters.
    *   **Example**:

        ```yaml title="config.yaml"
        trainer:
          callbacks:
            - _target_: lighter.callbacks.Freezer
              names: ["model.classifier.weight", "model.classifier.bias"] # Freeze classifier layer weights/bias
              until_step: 1000
        ```

        Config freezes parameters named `"model.classifier.weight"` and `"model.classifier.bias"` until step 1000.

3.  **Exclude Layers from Freezing (`except_names`, `except_name_starts_with`)**:

    *   Exclude layers from freezing (even if matched by `name_starts_with` or `names`) using `except_names`/`except_name_starts_with`.
    *   Selectively unfreeze parts of otherwise frozen module.
    *   **Example**:

        ```yaml title="config.yaml"
        trainer:
          callbacks:
            - _target_: lighter.callbacks.Freezer
              name_starts_with: ["model.encoder"]           # Freeze all encoder layers
              except_name_starts_with: ["model.encoder.layer5"] # Except "model.encoder.layer5" layers
              until_epoch: 7
        ```

        Config freezes `"model.encoder"` layers except `"model.encoder.layer5"`, keeping layer5 trainable.

4.  **Unfreezing after Condition (`until_step`, `until_epoch`)**:

    *   `until_step`: Unfreeze layers after training step.
    *   `until_epoch`: Unfreeze layers after epoch.
    *   Use either/both `until_step`/`until_epoch`. Unfreezes when either condition met.
    *   Omit `until_step`/`until_epoch` to freeze layers for entire training (or manual unfreezing).
    *   **Example**:

        ```yaml title="config.yaml"
        trainer:
          callbacks:
            - _target_: lighter.callbacks.Freezer
              name_starts_with: ["model.backbone"]
              until_epoch: 5    # Unfreeze after epoch 5
              until_step: 5000  # OR after step 5000
        ```

        Config unfreezes `"model.backbone"` layers after epoch 5 OR step 5000 (whichever first).

**Combining Freezing Strategies**:

Combine `Freezer` callbacks in `config.yaml` for complex freezing schedules. E.g., initial backbone freeze, gradual part unfreezing.

**Example: Gradual Layer Unfreezing**

```yaml title="config.yaml"
trainer:
  callbacks:
    # Unfreeze backbone layers at epoch 5
    - _target_: lighter.callbacks.Freezer
      name_starts_with: ["model.backbone"]
      until_epoch: 5
    # Unfreeze early encoder layers at epoch 10
    - _target_: lighter.callbacks.Freezer
      name_starts_with: ["model.encoder.layer1", "model.encoder.layer2"]
      until_epoch: 10
```

Example: 2 `Freezer` callbacks for gradual unfreezing - initial backbone freeze, gradual encoder layer unfreezing.

**Inspecting Frozen Layers**:

`Freezer` callback logs freezing info during training. Check logs (TensorBoard/console) to verify.

**`Freezer` Callback Use Cases**:

*   **Transfer Learning**: Freeze pre-trained model's early layers, train head.
*   **Fine-tuning**: Gradually unfreeze pre-trained layers.
*   **Training Stability**: Initial layer freezing.
*   **Regularization**: Layer freezing for regularization.
*   **Efficient Training**: Reduce training time/memory.

## Practical Example: Transfer Learning

```yaml
trainer:
  callbacks:
    # Stage 1: Train only the classifier head
    - _target_: lighter.callbacks.Freezer
      name_starts_with: ["model.backbone"]  # Freeze pretrained backbone
      until_epoch: 5

    # Stage 2: Fine-tune top layers
    - _target_: lighter.callbacks.Freezer
      name_starts_with: ["model.backbone.layer1", "model.backbone.layer2"]
      until_epoch: 10  # Keep early layers frozen longer
```

This progressive unfreezing strategy:
1. **Epochs 0-5**: Only train the classifier head
2. **Epochs 5-10**: Unfreeze and train top backbone layers
3. **Epochs 10+**: Train entire model

## Advanced Pattern: Discriminative Learning Rates

```yaml
# Combine freezing with different learning rates
trainer:
  callbacks:
    - _target_: lighter.callbacks.Freezer
      name_starts_with: ["model.backbone"]
      until_epoch: 5

system:
  optimizer:
    _target_: torch.optim.Adam
    params:
      - params: "$[p for n, p in @system::model.named_parameters() if 'backbone' in n]"
        lr: 0.0001  # Lower LR for pretrained layers
      - params: "$[p for n, p in @system::model.named_parameters() if 'head' in n]"
        lr: 0.001   # Higher LR for new layers
```

## Troubleshooting Guide 🔧

| Issue | Solution |
|-------|----------|
| **Parameters not freezing** | Print parameter names to verify: `for n, p in model.named_parameters(): print(n, p.requires_grad)` |
| **Performance drops after unfreezing** | Reduce LR with scheduler at unfreeze epoch |
| **BatchNorm issues** | Keep BN layers in eval mode even when unfrozen |
| **Memory increases** | Use gradient checkpointing or accumulation |
| **Don't know what to freeze** | Print model structure with `model.named_children()` |

## Best Practices 🏆

1. **Start Conservative**: Freeze more layers initially, then gradually unfreeze
2. **Monitor Metrics**: Track validation loss when layers unfreeze
3. **Use Warmup**: Apply learning rate warmup after unfreezing
4. **Reduce LR**: Lower learning rate when unfreezing pretrained layers
5. **Test First**: Verify which layers to freeze by printing model structure

## Quick Reference Card 📄

```yaml
# Freeze by prefix
name_starts_with: ["model.encoder", "model.embeddings"]

# Freeze specific layers
names: ["model.layer1.weight", "model.layer1.bias"]

# Exclude from freezing
except_names: ["model.encoder.final_layer.weight"]
except_name_starts_with: ["model.encoder.norm"]

# Unfreeze timing
until_epoch: 10        # After epoch 10
until_step: 1000       # After step 1000
# Both: unfreeze when EITHER condition is met
```

## Recap and Next Steps

✅ **You've Mastered:**

- Strategic layer freezing for transfer learning
- Progressive unfreezing techniques
- Troubleshooting common freezing issues
- Best practices for stable training

🎯 **Key Insights:**

- Freezing preserves pretrained knowledge
- Gradual unfreezing prevents catastrophic forgetting
- Monitor performance when changing freeze status
- Combine with appropriate learning rates

💡 **Pro Tip:** Log which layers are frozen/unfrozen at each epoch for reproducibility!

## Related Guides
- [Run Guide](run.md) - Training workflows
- [Configuration](configure.md) - Advanced config patterns
