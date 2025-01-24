SCHEMA = {
    "_meta_": {"type": "dict"},
    "_requires_": {"type": ["string", "list", "dict"]},
    "project": {"type": "string"},
    "vars": {"type": "dict"},
    "args": {
        "type": "dict",
        "schema": {
            "fit": {"type": "dict"},
            "validate": {"type": "dict"},
            "test": {"type": "dict"},
            "predict": {"type": "dict"},
            "lr_find": {"type": "dict"},
            "scale_batch_size": {"type": "dict"},
        },
    },
    "trainer": {"type": "dict"},
    "system": {
        "type": "dict",
        "schema": {
            "_target_": {"type": "string", "required": True},
            "model": {"type": "dict"},
            "criterion": {"type": "dict"},
            "optimizer": {"type": "dict"},
            "inferer": {"type": "dict"},
            "metrics": {
                "type": "dict",
                "schema": {
                    "train": {"type": ["list", "dict"]},
                    "val": {"type": ["list", "dict"]},
                    "test": {"type": ["list", "dict"]},
                },
            },
            "dataloaders": {
                "type": "dict",
                "schema": {
                    "train": {"type": "dict"},
                    "val": {"type": "dict"},
                    "test": {"type": "dict"},
                    "predict": {"type": "dict"},
                },
            },
            "adapters": {
                "type": "dict",
                "schema": {
                    "train": {"type": "dict"},
                    "val": {"type": "dict"},
                    "test": {"type": "dict"},
                    "predict": {"type": "dict"},
                },
            },
        },
    },
}
