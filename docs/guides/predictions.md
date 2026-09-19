# Prediction output and return values

Prediction produces scientific outputs; a configured writer exports them. Keep each output bound to the sample ID and the concrete checkpoint that produced it.

## Export CSV

The [quick start](../quickstart.md#fit-evaluate-and-export) has a complete runnable export. Its step returns these fields:

```python
def predict_step(self, batch, batch_idx):
    return {
        "id": batch["id"],
        "prediction": self(batch["x"]).squeeze(-1),
        "target": batch["target"],
    }
```

Its Trainer callback section includes:

```yaml
trainer:
  callbacks:
    - _target_: lighter.callbacks.CsvWriter
      path: "$@output_dir + '/predictions.csv'"
      keys: [id, prediction, target]
```

These are excerpts from the diagnostic. Every configured key must be present, with compatible batch lengths. Keep IDs in the dataset and return them unchanged; a row number is not a stable cross-run identity.

## Stream or retain native outputs

Writers preserve Trainer's native return policy:

```python
# Excerpts: model, loader and trainer are your existing native objects.
predictions = trainer.predict(model, dataloaders=loader, return_predictions=True)
trainer.predict(model, dataloaders=loader, return_predictions=False)
```

The first retains all native outputs as well as writing artifacts; the second streams artifacts without retaining the complete prediction list. In the CLI, use `--no-return-predictions`. Runner accepts the same native keyword. Some launch strategies cannot return predictions; use their native streaming path.

Writers use public batch callbacks and do not modify Lightning's private prediction buffers. Bounded-memory behavior still depends on your step and callbacks not retaining their own growing outputs.

## CSV fields and publication

CsvWriter serializes literal CSV fields without inferring numeric or missing-value types. IDs such as `0001` and labels such as `NA`/`NULL` survive, including quoting, Unicode and embedded newlines. Read text identities with `csv.DictReader` or disable inference in your chosen reader. CSV carries no typed schema; Python `None` becomes the standard empty field.

Finalization verifies matching writer headers, streams rank shards to a temporary sibling file and replaces the destination only after the complete merge succeeds. A failed merge preserves the previous published file and shards for diagnosis. An empty shard with a valid header is supported. Shards are expected to come from CsvWriter.

Distributed writers require shared storage. Rank-shard merging does not by itself establish global dataset order, unique IDs or correct distributed sampling; validate the final population. The specific tested profiles are in [compatibility](compatibility.md#exact-validation-profiles).

## Save one file per sample

FileWriter writes a chosen output value with a registered or supplied callable. For example, this callback-section excerpt saves tensor outputs using the returned `id` as the sample name:

```yaml
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      directory: outputs/tensors
      value_key: prediction
      name_key: id
      writer_fn: tensor
```

Choose a serialization appropriate to the output. Built-in `image_2d` expects CHW values, clamps them to [0, 1] and scales to uint8 PNG. Passing raw uint8 [0, 255] values would saturate most nonzero pixels; normalize them first. This is not a spatial medical-volume export policy. Keep any required geometry, class meanings or preprocessing metadata with the artifacts.

For custom publication, use a native prediction callback or the documented [BaseWriter.write](../reference/callbacks/base_writer.md) interface. See [FileWriter](../reference/callbacks/file_writer.md) and [CsvWriter](../reference/callbacks/csv_writer.md) for arguments.

## Migration

Earlier writers could clear private Trainer storage and leave only the last batch in returned predictions. Set `return_predictions=False` explicitly when streaming; preserve returns when you need them. Readers that relied on pandas inference during finalization should apply intentional conversions when loading the artifact.
