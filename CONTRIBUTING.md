[![Apache 2.0][apache-shield]][apache] 

[apache]: https://opensource.org/licenses/Apache-2.0
[apache-shield]: https://img.shields.io/badge/License-Apache_2.0-blue.svg

Thank you for your interest in contributing to the BibMon project!

We expect to receive various types of contributions from individuals, research institutions, startups and companies.

In this guide we present how the expected contributions might be proposed.

## Getting started

The recommended first step is to read the project's [README](https://github.com/petrobras/BibMon/blob/main/README.md) for an overview of what this repository contains.

## Asking questions

Please do not open issues to ask questions. Please use the Discussions section accessed through the link that appears in the top menu.

## Before contributing

Before you can contribute to this project, we require you read and agree to the following documents:

* [CODE OF CONDUCT](https://github.com/petrobras/BibMon/blob/main/CODE_OF_CONDUCT.md);
* [CONTRIBUTOR LICENSE AGREEMENT](https://github.com/petrobras/BibMon/blob/main/CONTRIBUTOR_LICENSE_AGREEMENT.md);
* This contributing guide.

It is also very important to know, participate and follow the discussions. Click on the Discussions link that appears in the top menu.

## BibMon architecture

`BibMon` was designed with easy extensibility and maintenance in mind, even for users with little experience in software development. This was achieved through the use of the object-oriented paradigm, which allows for the pre-implementation of generic functionalities and enables new features to be quickly programmed by leveraging the original structure.

The following bulletpoints describe the library's architecture and provide instructions for implementing new functionalities.

* Models
   * `GenericModel`
      * [Abstract class](https://en.wikipedia.org/wiki/Abstract_type) that gathers all the common functionality for various monitoring models and should be used as a basis for implementing new models through inheritance.
      * Since it cannot be instantiated, it is not available in the library's namespace for direct import.
* Preprocessing techniques
   * `PreProcess`
      * Class that contains preprocessing methods (e.g., `normalize`, `remove_nan_observations`, etc.).
      * The `apply` method is responsible for sequentially applying the methods specified during the initialization of a `PreProcess` object.
* Data
   * Each dataset has its corresponding directory, such as `bibmon/tennessee_eastman` or `bibmon/real_process_data`.
   * The functions for loading the data are in the `_load_data.py` file.
* Alarms
   * Alarms are implemented as functions in the `_alarms.py` file and are stored by the `GenericModel` in the `self.alarms` dictionary.
* Additional Features
   * Additional features such as generating comparative tables, correlation analysis, etc., are programmed in the `_bibmon_tools.py` file.

## Implementing New Functionalities

When implementing new functionalities, don't forget to:

* Add the classes and functions to the library's namespace (in the `__init__.py` file).
* Document the classes and functions using docstrings in the [NumPy format](https://numpydoc.readthedocs.io/en/latest/format.html).

### Models

To implement a new model, create a .py file with the following import statement:

```python
from ._generic_model import GenericModel
```

In this file, create the class corresponding to the new model, inheriting from `GenericModel`:

```python
class NewModel(GenericModel):
```

Implement two methods in `NewModel`:

* `train_core`: a function that prepares the model for making predictions.
* `map_from_X`: a function that receives a dataset `X` and returns the predicted dataset (`Y_pred`) or reconstructed dataset (`X_pred`).

The functionality implemented in `train_core` is specific to the considered model since the generic preprocessing steps are already executed in the `pre_training` method of `GenericModel`.

Optionally, implement an `__init__()` constructor or any other necessary methods.

### Preprocessing Techniques

Preprocessing techniques should be implemented as methods of the `PreProcess` class following the pattern:

```python
def new_method(self, df, train_or_test='train'):
    if self.is_Y:
        # code generating a processed df if the data is Y
        if train_or_test == 'train':
            # code generating a processed df if the data is for training
    if ...
    return processed_df
```

Use the `self.is_Y` and `train_or_test` flags to cover the possibilities regarding the nature of the data in the model (predictors `X` or predicted `Y`) and the analysis stage (training or testing).

Even if not used, the `train_or_test` flag should be present in the input parameters.

### Data

Add the new dataset to a specific subdirectory in the `bibmon` directory. Program the data loading function in the `_load_data.py` file.

### Alarms

To program a new alarm logic, define a function in the `_alarms.py` file.

To apply the new logic in the library, it will be necessary to implement the use of the functionality in the methods of the `GenericModel` class by creating a new key in the `self.alarms` dictionary. If necessary, window sizes and other parameters should be programmed in the method entries.

### Additional Features

Preferably, use the `_bibmon_tools.py` file to implement additional features.