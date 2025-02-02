"""
Defines the schema for configuration validation in the Lighter framework.

The schema ensures user configurations are correctly structured and typed. It includes:
- `_meta_`: Metadata as a dictionary.
- `_requires_`: Runs first, primarily to be used for imports.
- `project`: Project name as a string.
- `vars`: Variables as a dictionary.
- `args`: Arguments to pass to Trainer stage methods like `fit`, `validate`, `test`.
- `trainer`: Trainer setup.
- `system`: System setup, encapsulates model, criterion, optimizer, scheduler, inferer, metrics, dataloaders, and adapters.

Used by the `Config` class for validation.
"""

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
    "trainer": {"type": "dict", "required": True},
    "system": {
        "type": "dict",
        "schema": {
            "_target_": {"type": "string", "required": True},
            "model": {"type": "dict"},
            "criterion": {"type": "dict"},
            "optimizer": {"type": "dict"},
            "scheduler": {"type": "dict"},
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
                    "train": {
                        "type": "dict",
                        "schema": {
                            "batch": {"type": "dict"},
                            "criterion": {"type": "dict"},
                            "metrics": {"type": "dict"},
                            "logging": {"type": "dict"},
                        },
                    },
                    "val": {
                        "type": "dict",
                        "schema": {
                            "batch": {"type": "dict"},
                            "criterion": {"type": "dict"},
                            "metrics": {"type": "dict"},
                            "logging": {"type": "dict"},
                        },
                    },
                    "test": {
                        "type": "dict",
                        "schema": {"batch": {"type": "dict"}, "metrics": {"type": "dict"}, "logging": {"type": "dict"}},
                    },
                    "predict": {"type": "dict", "schema": {"batch": {"type": "dict"}, "logging": {"type": "dict"}}},
                },
            },
        },
    },
}
