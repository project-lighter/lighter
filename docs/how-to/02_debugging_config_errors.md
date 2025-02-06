# How to Debug Configuration Errors in Lighter

## Introduction to Debugging Config Errors

Configuration files are central to Lighter for declarative and reproducible experiments. Misconfigurations in `config.yaml` are common, especially for new users or complex setups.

This guide provides a systematic approach to debug Lighter configuration errors, helping you quickly identify and resolve `config.yaml` issues. Following these steps and understanding common error types will streamline debugging and ensure smooth Lighter experiments.

## Common Types of Configuration Errors

Configuration errors in Lighter can be broadly categorized into the following types:

1.  **YAML Syntax Errors**: These errors arise from incorrect YAML syntax in your `config.yaml` file, such as:
    *   Incorrect indentation (YAML is indentation-sensitive).
    *   Missing colons, hyphens, or other YAML syntax elements.
    *   Invalid YAML structures (e.g., incorrect nesting of lists or dictionaries).
    *   Unescaped special characters.

    **Example: Common YAML Syntax Errors**

    ```yaml title="config.yaml"
    system: 
      model: # Correct indentation for 'model'
        _target_: my_project.models.MyModel 
    optimizer: # Missing colon after 'optimizer' - YAML syntax error
    _target_: torch.optim.Adam # Incorrect indentation for '_target_'
      lr: 0.001
    ```

    In this example, the `optimizer` key is missing a colon, and `_target_` is incorrectly indented under `optimizer` due to the missing colon, both causing YAML syntax errors.

2.  **Validation Errors**: Lighter uses Cerberus, a powerful validation library, to validate your `config.yaml` file against a predefined schema. Validation errors occur when your configuration does not conform to this schema, such as:
    *   Missing required fields.
    *   Incorrect data types for fields (e.g., string instead of integer).
    *   Invalid values for fields (e.g., value outside allowed range).
    *   Incorrect structure of nested configurations.

    **Example: Common Validation Errors**

    ```yaml title="config.yaml"
    trainer:
      max_epochs: 100 # Correct data type

    system:
      optimizer:
        _target_: torch.optim.Adam
        lr: 0.001
      model: 
        _target_: my_project.models.MyModel
        params: # Nested configuration for model parameters
          input_size: 28 # Correct data type - integer
          num_classes: "10" # num_classes should be integer, not string - Validation error in nested config
    ```

    In this example, `trainer.max_epochs` is corrected to an integer. However, within the nested `system.model.params` configuration, `num_classes` is incorrectly set to a string `"10"` instead of an integer, causing a validation error in a nested configuration.

3.  **Import Errors**: These errors occur when Lighter fails to dynamically import modules or classes specified in your `config.yaml` using the `_target_` key. Common causes of import errors include:
    *   Incorrect module paths in `_target_` specifications.
    *   Typos in module or class names.
    *   Missing `project` path specification in `config.yaml` when using custom modules.
    *   Issues with your project's Python environment or module dependencies.
    *   Circular dependencies in your custom modules.

    **Example: Common Import Errors**

    ```yaml title="config.yaml"
    project: my_project/

    system:
      model:
        _target_: my_project.models.MyyModel # Typo in class name 'MyyModel' - Import error (class not found)

    trainer:
      callbacks:
        - _target_: lighter.callbacks.MyCallback # Custom callback not in 'lighter.callbacks' - Import error (module not found)
    ```

4.  **Runtime Errors**: Even if your `config.yaml` passes YAML syntax and validation checks, and modules are imported successfully, you may still encounter runtime errors during experiment execution. These errors can arise from:
    *   Incorrect arguments passed to classes or functions in your configuration.
    *   Logical errors in your custom modules' code.
    *   Incompatibility issues between configured components.
    *   Resource limitations (e.g., out-of-memory errors).

    **Example: Common Runtime Errors**

    ```yaml title="config.yaml"
    system:
      model:
        _target_: my_project.models.MyModel
        # ... model params ...

      dataloaders:
        train:
          _target_: torch.utils.data.DataLoader
          dataset:
            _target_: my_project.datasets.MyDataset
            data_path: "data/train_data.csv" # Assuming dataset expects 'data_path'
          batch_size: 32
          # ... 

      optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()"
        lr: 0.001
    ```

    If `my_project.datasets.MyDataset` expects the data path to be `"data/training_data.csv"` but the config provides `"data/train_data.csv"`, it will lead to a runtime error when the data loader tries to access or process the file, as the file path in the config is incorrect. This is a common scenario of runtime errors due to configuration mismatches.

## Systematic Debugging Steps

To effectively debug configuration errors, follow these steps:

1.  **Read Error Messages Carefully**: Lighter error messages provide clues about the error type and location.

    *   **Identify Error Type**: YAML syntax, validation, import, or runtime error.
    *   **Locate Error Source**: Check line number/section in `config.yaml`, module/class indicated.
    *   **Understand Root Cause**: Determine if it's syntax, validation, import, or runtime issue.

2.  **Validate YAML Syntax**: Review `config.yaml` for YAML syntax errors.

    *   **Indentation**: Check for consistent indentation (2 spaces).
    *   **Colons/Hyphens**: Look for missing colons/hyphens.
    *   **Quotes**: Verify correct string quoting.
    *   **YAML Linters**: Use online linters or VS Code extensions.

3.  **Validate Against Schema**: For validation errors, check `config.yaml` against Lighter's schema in `engine/config.py`.

    *   **Refer to `SCHEMA`**: Examine `lighter/engine/config.py` for schema definition.
    *   **Compare to `SCHEMA`**: Check `config.yaml` against `SCHEMA` for discrepancies.
    *   **Nested Configurations**: Pay attention to nested config structures.

4.  **Troubleshoot Import Errors**: Systematically debug module import issues.

    *   **Verify Module Paths**: Check `_target_` paths in `config.yaml`.
    *   **Check for Typos**: Review module/class names, file paths for typos.
    *   **Ensure `project` Path Correct**: Verify `project` path in `config.yaml` (for custom modules).
    *   **Verify Module Files Exist**: Ensure module files exist at specified paths.
    *   **Check Python Environment**: Verify Python environment and dependencies.
    *   **Inspect Error Stack Traces**: Examine stack traces for import failure details.

5.  **Debug Runtime Errors**: Debug runtime errors during experiment execution.

    *   **Examine Error Stack Traces**: Pinpoint error line from stack trace.
    *   **Check Argument Compatibility**: Verify argument compatibility in `config.yaml`.
    *   **Inspect Custom Module Code**: Review custom module code for logic errors.
    *   **Use Python Debugger**: Use debuggers (e.g., `pdb`, VS Code debugger) to step through code.
    *   **Simplify Configuration**: Simplify `config.yaml` to isolate errors.

6.  **Seek Community Help**: If unresolved, seek help from the Lighter community.

    *   **Consult Documentation**: Review Lighter documentation.
    *   **Check GitHub Repository**: Explore Lighter GitHub for FAQs, issues.
    *   **Reach Out to Community**: Post error details in Lighter forums.

## Example Scenarios and Solutions

To further illustrate the debugging process, let's consider a few example scenarios of common configuration errors and their solutions:

**Scenario 1: Validation Error - Missing Required Field**

**Error Message**:

```
Cerberus validation error in section 'system.model':
Missing required field: '_target_'
```

**Cause**: The `model` section in the `config.yaml` is missing the required `_target_` field, which specifies the model class to be used.

**Solution**: Add the `_target_` field to the `model` section, specifying the desired model class:

```yaml title="config.yaml"
system:
  model:
    _target_: my_project.models.MyModel # Add '_target_' field
    # ... model arguments ...
```

**Scenario 2: Import Error - Module Not Found**

**Error Message**:

```
ImportError: ModuleNotFoundError: No module named 'my_project.mmodels'
```

**Cause**: There is a typo in the module path `my_project.mmodels` in the `_target_` specification. The correct module path is likely `my_project.models`.

**Solution**: Correct the typo in the module path in the `_target_` specification:

```yaml title="config.yaml"
system:
  model:
    _target_: my_project.models.MyModel # Correct module path to 'my_project.models'
    # ... model arguments ...
```

**Scenario 3: Runtime Error - Invalid Argument Type**

**Error Message**:

```
TypeError: __init__() got an unexpected keyword argument 'input_size'
```

**Cause**: The `MyModel` class constructor (`__init__`) in `my_project.models.MyModel` does not accept an argument named `input_size`, but the `config.yaml` is providing this argument.

**Solution**:

1.  **Inspect the `MyModel` Class**: Examine the `my_project/models/my_model.py` file to check the constructor (`__init__`) definition of the `MyModel` class and identify the correct argument names it accepts.
2.  **Correct Argument Names in `config.yaml`**: Update the `config.yaml` file to use the correct argument names as defined in the `MyModel` class constructor.

    For example, if the `MyModel` class constructor expects `input_dim` instead of `input_size`, update the `config.yaml` as follows:

    ```yaml title="config.yaml"
    system:
      model:
        _target_: my_project.models.MyModel
        input_dim: 784 # Correct argument name to 'input_dim'
        # ... other arguments ...
    ```

## Recap: Master Config Debugging

Mastering config debugging is key to efficient Lighter use. Understanding error types and using systematic steps streamlines issue resolution, enabling smoother workflows and focused research.

Next, explore the [How-To guide on Creating a Custom Metric](03_creating_a_custom_metric.md) to learn how to extend Lighter with your own evaluation metrics, or return to the [How-To guides section](../how-to/01_custom_project_modules.md) for more practical problem-solving guides. You can also go back to the [Design section](../design/01_overview.md) for more conceptual documentation or the [Tutorials section](../tutorials/01_configuration_basics.md) for end-to-end examples.
