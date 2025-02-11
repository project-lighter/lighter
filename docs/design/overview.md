## Challenges of Deep Learning Experimentation

Deep learning has revolutionized many fields, but the process of conducting deep learning research is still complex and time-consuming. Practitioners often face challenges such as:

<div class="grid cards" markdown>

-   :material-clock-time-ten:{ .lg .middle } __Boilerplate Code__

    ---

    Writing repetitive code for training loops, data loading, metric calculation, and experiment setup for each new project.

-   :material-text-search:{ .lg .middle } __Experiment Management__

    ---

    Managing a range of hyperparameters, datasets, and architectures across different experiments.

-   :material-refresh:{ .lg .middle } __Reproducibility__

    ---

    Ensuring that experiments are reproducible and that results can be reliably replicated.


-   :material-account-group:{ .lg .middle } __Collaboration__

    ---

    Sharing and collaborating on experiments with other researchers or team members.

</div>

## Lighter: Configuration-Driven Deep Learning

**Lighter** is a framework designed to address these challenges and streamline deep learning experimentation. It uses YAML configuration files for experiment definition and management.

### Core Philosophy

Lighter's core philosophy is to **separate configuration from code** and provide a **high-level, declarative way to define deep learning experiments**. This is achieved through the following key design principles:

1.  **Declarative Experiment Definition with YAML:** Lighter embraces a declarative approach, where experiments are defined entirely through YAML configuration files. This contrasts with an imperative approach, where the experiment logic is written directly in code. The YAML configuration acts as a blueprint, specifying *what* components to use and *how* they are connected, without dictating the *implementation details* of the training loop or data processing. This separation of concerns is fundamental to Lighter's design.

2.  **Component-Based Architecture:** Lighter structures deep learning experiments into distinct, well-defined components:

    *   **`model`:** The neural network architecture.
    *   **`optimizer`:** The optimization algorithm (e.g., Adam, SGD).
    *   **`scheduler`:** Learning rate scheduler (optional).
    *   **`criterion`:** The loss function.
    *   **`metrics`:** Evaluation metrics (e.g., accuracy, F1-score).
    *   **`dataloaders`:** Data loading and preprocessing pipelines.
    *   **`inferer`:** (Optional) Handles inference logic, like sliding window inference.
    *    **`adapters`**: Customize data handling, argument passing, and transformations.

    Each component is defined in in the `system` section of the YAML configuration. This `system` then interacts with PyTorch Lightning's `Trainer` to orchestrate the training process.

3.  **Encapsulation and Orchestration with `System`:** The `lighter.System` acts as a central orchestrator, encapsulating all the experiment's components. It inherits from PyTorch Lightning's `LightningModule`, providing a familiar and well-defined interface for training, validation, testing, and prediction.  Crucially, the `System` class is responsible for:

    *   Instantiating the components defined in the YAML configuration.
    *   **Connecting these components and managing the flow of data between them.**  This includes passing the model's parameters to the optimizer, feeding data through the model, calculating the loss, and updating metrics.  The `Adapters` play a vital role in this data flow management.
    *   Providing hooks for custom logic (e.g., `on_train_start`).
    *   Defining the `forward` method, which is used for inference, and delegating to a custom `inferer` if one is specified.
    *   Implementing the `_step` method, common to all stages (`training_step`, `validation_step`, `test_step`, `predict_step`), which handles batch preparation, forward pass, loss calculation, and metric computation.

    The `System` class *does not* implement the training loop itself. Instead, it defines *how* the components interact within each step of the loop.

4.  **Leveraging PyTorch Lightning's `Trainer`:** Lighter builds upon PyTorch Lightning's `Trainer`, inheriting its powerful features for distributed training, mixed precision, checkpointing, and, most importantly, the **training loop logic**. The `trainer` section in the YAML configuration allows users to directly configure the `Trainer` object, providing access to all of PyTorch Lightning's capabilities.  The `Trainer` executes the training loop, calling the `System`'s methods (e.g., `training_step`, `validation_step`) at each iteration. This separation of concerns keeps Lighter's core logic focused on experiment configuration and component management.

5.  **Project-Specific Custom Modules:** Lighter is designed to be highly extensible. Users can define custom modules (models, datasets, metrics, callbacks, etc.) within their own project directory and seamlessly integrate them into Lighter experiments. The `project` key in the YAML configuration specifies the path to the project's root directory. Lighter then dynamically imports these custom modules, making them available for use in the configuration file via the `_target_` key. This eliminates the need to modify Lighter's core code to add custom functionality, promoting flexibility and code organization.

6. **Adapters: The Key to Task-Agnosticism**: Lighter's `Adapters` are a crucial design element that enables Lighter to be task-agnostic. Adapters provide a way to customize the flow of data between different components *without* modifying the components themselves. They act as intermediaries, handling variations in data formats, argument orders, and pre/post-processing requirements. Lighter provides several types of adapters:

    *   **`BatchAdapter`:**  Handles variations in batch structure.  Different datasets or tasks might return batches in different formats (e.g., `(input, target)`, `input`, `{"image": input, "label": target}`). The `BatchAdapter` allows users to specify how to extract the `input`, `target`, and `identifier` (e.g., filename) from the batch, regardless of its original structure. This uses accessor functions, integer indices, or string keys, as appropriate.

    *   **`CriterionAdapter` and `MetricsAdapter`:**  Handle variations in the argument signatures of loss functions and metrics. Some loss functions might expect arguments in the order `(pred, target)`, while others might expect `(target, pred)` or even include the `input`. These adapters allow users to specify the correct argument mapping (positional or keyword) and apply transformations (e.g., `torch.sigmoid`) *before* the data is passed to the criterion or metric. This avoids the need to write wrapper functions for each loss function or metric.

    *   **`LoggingAdapter`:**  Handles transformations applied to data *before* it is logged. This is useful for tasks like converting predictions to class labels (using `torch.argmax`) or converting tensors to NumPy arrays for visualization.

    The adapter system allows Lighter to handle a wide range of tasks (classification, segmentation, self-supervised learning, etc.) without requiring changes to the core framework.  It provides a powerful and flexible mechanism for customizing data flow, making Lighter highly adaptable to different research problems.

## Recap

*   **Declarative Configuration:** Defining experiments through YAML files.
*   **Modularity:**  Breaking down experiments into well-defined components.
*   **Encapsulation and Orchestration**: Using the `System` class to manage component interactions and data flow.
*   **Extensibility:**  Allowing users to easily incorporate custom modules.
*   **Flexibility:**  Adapting to diverse tasks and data formats through the adapter system.
*    **Simplicity**: Leveraging PyTorch Lightning for robust training features, allowing the user to run their experiment with a single command.
