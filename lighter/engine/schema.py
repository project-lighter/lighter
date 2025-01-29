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
            "scale_batch_size": {"type": "dict"}
        }
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
                    "test": {"type": ["list", "dict"]}
                }
            },
            "dataloaders": {
                "type": "dict",
                "schema": {
                    "train": {"type": "dict"},
                    "val": {"type": "dict"},
                    "test": {"type": "dict"},
                    "predict": {"type": "dict"}
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
                            "logging": {"type": "dict"}
                        }
                    },
                    "val": {
                        "type": "dict",
                        "schema": {
                            "batch": {"type": "dict"},
                            "criterion": {"type": "dict"},
                            "metrics": {"type": "dict"},
                            "logging": {"type": "dict"}
                        }
                    },
                    "test": {
                        "type": "dict",
                        "schema": {
                            "batch": {"type": "dict"},
                            "metrics": {"type": "dict"},
                            "logging": {"type": "dict"}
                        }
                    },
                    "predict": {
                        "type": "dict",
                        "schema": {
                            "batch": {"type": "dict"},
                            "logging": {"type": "dict"}
                        }
                    }
                }
            }
        }
        }
}
