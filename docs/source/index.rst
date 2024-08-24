.. BibMon documentation master file, created by
   sphinx-quickstart on Wed Jul 27 12:04:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BibMon: Process Monitoring Library
==================================

BibMon (from the Portuguese Biblioteca de Monitoramento de Processos, or Process Monitoring Library) is a Python package that provides deviation-based predictive models for fault detection, soft sensing, and process condition monitoring.

Features
----------------------

The resources offered by `BibMon` are:

* Application in online systems: a trained `BibMon` model can be used for online analysis with both individual samples and data windows. For each sample or window, a prediction is made, the model state is updated, and alarms are calculated.
* Compatibility, within the same architecture, of regression models (i.e., virtual sensors, containing separate X and Y data, such as RandomForest) and reconstruction models (containing only X data, such as PCA).
* Preprocessing pipelines that take into account the differences between X and Y data and between training and testing stages.
* Possibility of programming different alarm logics.
* Easy extensibility through inheritance (there is a class called `GenericModel` that implements all the common functionality for various models and can be used as a base for implementing new models). For details, consult the `CONTRIBUTING.md` file.
* Convenience functions for performing automatic offline analysis and plotting control charts.
* Real and simulated process datasets available for importing.
* Comparative tables to automate the performance analysis of different models.
* Automatic hyperparameter tuning using Optuna.

Getting started
-----------------------------

.. toctree::
   :maxdepth: 3

   install
   usage
   tutorials
   sci_article

Contributing
----------------------

BibMon is an open-source project driven by the community. If you would like to contribute to the project, please refer to the following contributing page.

.. toctree::
   :maxdepth: 2

   contributing.md

The API Documentation
-----------------------------

In this section you will find information about specific functions, classes, or methods.

.. toctree::
   :maxdepth: 3

   api
