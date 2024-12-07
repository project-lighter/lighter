# Inferer

The inferer in Lighter is used for making predictions on data. It is typically used in validation, testing, and prediction workflows.

## Using Inferers
Inferers must be classes with a `__call__` method that accepts two arguments: the input to infer over and the model itself. They are used to handle complex inference scenarios, such as patch-based or sliding window inference.

## MONAI Inferers
Lighter integrates with MONAI inferers, which cover most common inference scenarios. You can use MONAI's sliding window or patch-based inferers directly in your Lighter configuration.

For more information on MONAI inferers, visit the [MONAI documentation](https://docs.monai.io/en/stable/inferers.html).
