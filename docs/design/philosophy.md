# Design Philosophy

Why Lighter is designed the way it is.

## Core Principles

### 1. Configuration Over Code
Experiments are data, not code. YAML configs are easier to:
- Version control and compare
- Share and reproduce
- Parametrize and sweep
- Audit and validate

### 2. Composition Over Inheritance
Instead of subclassing for behavior changes, compose with Flows. More flexible, less coupled.

### 3. Convention Over Configuration
Sensible defaults (like BatchAdapter assuming `(input, target)` tuples) reduce boilerplate. Override when needed.

### 4. Separation of Concerns
Clear boundaries:
- **Config** - Experiment definition
- **System** - Component orchestration
- **Trainer** - Execution engine
- **Flows** - Interface translation

### 5. Task-Agnostic by Design
No per-task pipelines. Flows handle variability, enabling unlimited flexibility for novel research.

## Why ~1,000 Lines of Code?

**Benefits:**
- Read entire codebase in an afternoon
- Easy to debug and understand
- Simple to extend and maintain
- Low long-term maintenance burden

**Achieved by:**
- Leveraging PyTorch Lightning (not reinventing training loops)
- Using MONAI's config system (proven, robust)
- Focusing on core value: config-driven experiments + adapters

## Integration Philosophy

### Standing on Shoulders of Giants

**PyTorch Lightning** - Battle-tested training engine
- Multi-GPU/TPU support
- Callbacks, loggers, profilers
- Gradient accumulation, mixed precision
- [→ PL Trainer docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html)

**MONAI** - Proven configuration system
- Config parsing and validation
- Reference resolution
- Dynamic instantiation
- [→ MONAI config docs](https://docs.monai.io/en/stable/config_syntax.html)

Lighter adds: Flows + System orchestration.

## Trade-offs

### When to Use Lighter

- Configuration-driven experiments are valuable
- You need task-agnostic flexibility
- You want minimal framework overhead
- Reproducibility and sharing are priorities

### When NOT to Use Lighter

- Highly custom training loops (use PyTorch directly)
- Prefer code over configuration
- Need high-level AutoML (use Ludwig)
- Domain-specific pipelines sufficient (use GaNDLF/Quadra)

## Learn More

- [Architecture Overview](overview.md) - Component details
- [Flows](flows.md) - Deep dive
- [Configuration Guide](../how-to/configure.md) - Practical usage
