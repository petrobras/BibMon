Usage
========================

Essentially, the library is used in two steps:

1. In the training step, a model is generated that captures the relationships between variables in the normal process condition;
2. In the prediction step, process data is compared to the model's predictions, resulting in deviations; if these deviations exceed a predefined limit, alarms are triggered.

Specifically, the implemented control charts are based on squared prediction error (SPE).

Follow the tutorials available in this documentation for a demonstration of BibMon usage for different scenarios.