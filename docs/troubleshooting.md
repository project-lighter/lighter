---
title: Troubleshooting
---

# Troubleshooting

Two switches help with any problem:

- `--print_config` prints the composed config, after all files and overrides, without building anything.
- `LIGHTER_DEBUG=1 lighter fit config.yaml` shows the full traceback, framework frames included.

Lighter exits with code 2 when the config cannot be built, and with 1 when your own code raised while it was
being built. Errors during the stage itself are Lightning's and pass through unchanged.
