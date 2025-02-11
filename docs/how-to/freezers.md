Lighter's `Freezer` callback specifies which model layers/parameters to freeze/unfreeze during training, and duration. For example, to freeze the encoder layers of a pre-trained model while training the classifier head or decoder

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.Freezer # Use the Freezer callback
      name_starts_with: ["model.encoder"] # Freeze layers starting with "model.encoder"
      until_epoch: 10                   # Unfreeze after epoch 10
```

*   **`_target_: lighter.callbacks.Freezer`**: `Freezer` callback.
*   **`name_starts_with`**: Freeze layers starting with prefixes (ResNet-18 conv layers).
*   **`until_epoch: 10`**: Unfreeze layers after epoch 10.

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
    - _target_: lighter.callbacks.Freezer # Freezer callback for initial freezing
      name_starts_with: ["model.backbone"] # Freeze backbone initially
      until_epoch: 5                     # Unfreeze backbone after epoch 5
    - _target_: lighter.callbacks.Freezer # 2nd Freezer callback for gradual unfreezing
      name_starts_with: ["model.encoder.layer1", "model.encoder.layer2"] # Gradually unfreeze layer1/layer2
      start_epoch: 5                     # Start unfreezing from epoch 5
      unfreeze_every_n_epochs: 5        # Unfreeze every 5 epochs
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

## Recap and Next Steps

Lighter `Freezer` callback: flexible, fine-grained model layer training control via `config.yaml`. Optimize training, performance, pre-trained knowledge.

Next: [Custom Inferer How-To guide](07_using_inferers.md), [How-To guides](../how-to/01_custom_project_modules.md), [Design section](../design/01_overview.md), [Tutorials section](../tutorials/01_configuration_basics.md).
