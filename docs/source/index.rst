================
What is Compose?
================

.. toctree::
    :hidden:
    :maxdepth: 1

    install
    getting_started
    main_concepts
    user_guide
    examples
    api_reference
    faq
    changelog

------------

|

.. image:: images/compose.png
    :width: 500px
    :align: center

|

**Compose** is a machine learning tool for automated prediction engineering. It allows you to structure prediction problems and generate labels for supervised learning. An end user defines an outcome of interest by writing a *labeling function*, then runs a search to automatically extract training examples from historical data. Its result is then provided to Featuretools_ for automated feature engineering and subsequently to EvalML_ for automated machine learning. The workflow of an applied machine learning engineer then becomes:

.. _Featuretools: https://docs.featuretools.com/
.. _EvalML: https://evalml.alteryx.com/

|

.. image:: images/workflow.png
    :align: center

|

By automating the early stage of the machine learning pipeline, our end user can easily define a task and solve it.

.. include:: main_concepts.rst
