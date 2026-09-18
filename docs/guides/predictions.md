# Prediction output and return values

Lighter writers persist predictions through Lightning's public batch callback. They preserve Trainer's return policy:

```python
# Retain complete native predictions as well as any configured artifacts.
predictions = trainer.predict(model, dataloaders=loader, return_predictions=True)

# Stream artifacts without retaining all batch outputs in memory.
trainer.predict(model, dataloaders=loader, return_predictions=False)
```

The same keyword can be passed through `Runner.run(Stage.PREDICT, inputs, return_predictions=False)`. Leaving it unspecified follows the selected Lightning strategy's native default. Some launch strategies do not support returning predictions; use native streaming behavior for those strategies. Writers never modify Lightning's private prediction buffers. Keeping returns enabled intentionally retains outputs in addition to writing them.

## CSV fields and publication

`CsvWriter` serializes configured fields and merges rank shards as CSV records without inferring numeric or missing-value types. Identifiers such as `0001` and literal labels such as `NA` and `NULL` survive finalization. Quoting, Unicode and embedded newlines retain their CSV field meaning. A consumer that wants text identities should also disable type/missing-value inference or use `csv.DictReader`. CSV itself does not carry a typed schema; Python None follows the standard CSV writer's empty-field behavior.

Finalization streams records instead of loading the full export into DataFrames. The batch writer validates configured fields and lengths; finalization verifies each shard's matching writer header, copies serialized records (including long text fields) without a CSV reader limit, writes a temporary file beside the destination, and replaces the destination only after the complete merge succeeds. A failed merge preserves the previous published file and rank shards for diagnosis. An empty shard with a valid header is supported. Finalization assumes shards produced by CsvWriter. Distributed writers require shared storage; these semantics do not imply that all hardware/launcher configurations have been tested.

Migration: earlier writers cleared private Trainer storage and could return only the last batch even when complete returns were requested. Set `return_predictions=False` explicitly for bounded-memory streaming. Code relying on pandas inference during finalization should apply its intended conversions when reading the artifact instead.
