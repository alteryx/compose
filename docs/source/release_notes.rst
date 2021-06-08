=============
Release Notes
=============

|

**Future Release**
    * Enhancements
        * Add ``maximum_data`` parameter to control when a search should stop (:pr:`216`)
    * Fixes
    * Documentation Changes
    * Testing Changes
        * Check if release notes were updated (:pr:`217`)
        * Add minimum dependency checker to generate minimum requirement files (:pr:`218`)
        * Add CI workflow for unit tests with minimum dependencies (:pr:`220`)
        * Create separate worksflows for each CI job (:pr:`220`)
        * Pass token to authorize uploading of codecov reports (:pr:`226`)

    | Thanks to the following people for contributing to this release:
    | :user:`gsheni`, :user:`jeff-hernandez`

|

**v0.6.0** February 11, 2021
    * Enhancements
        * Added description for continuous target distributions (:pr:`187`)
    * Fixes
        * Sorted label distribution in description (:pr:`188`)
    * Documentation Changes
        * Made logo easier to read (:pr:`182`)
        * Added Alteryx footer to docs (:pr:`185`, :pr:`186`)
        * Updated tutorials to the latest API changes (:pr:`190`, :pr:`198`, :pr:`204`)
        * Updated repository links to GitHub (:pr:`191`)
        * Added help page to docs (:pr:`194`)
        * Improved docs based on tech writer feedback (:pr:`195`)
        * Added open graph info to docs (:pr:`203`)
    * Testing Changes
        * Migrated CI tests to Github Actions (:pr:`184`, :pr:`189`)
        * Updated tests to trigger on pull request events (:pr:`199`)

    | Thanks to the following people for contributing to this release:
    | :user:`flowersw`, :user:`jeff-hernandez`, :user:`rwedge`

|

**v0.5.1** September 22, 2020
    * Documentation Changes
        * Update F1 Macro in Turbofan Degradation Tutorial (:pr:`180`).
        * Apply Pandas Docs Theme (:pr:`172`).
        * Add Chicago Bike Tutorial (:pr:`157`).
    * Testing Changes
        * Test Doc Builds (:pr:`165`)

|

**v0.5.0** August 28, 2020
    * Enhancements
        * Added Column-Based Windows (:pr:`151`).
    * Changes
        * Refactored Data Slice Generator (:pr:`150`).
    * Documentation Changes
        * Updated README (:pr:`164`).
        * Updated Predict Next Purchase Demo (:pr:`154`).
        * Updated Predict Turbofan Degradation Demo (:pr:`154`).

.. warning::

    **Breaking Changes**
        * Attributes of the data slice context have changed. Inside a labeling function, the timestamps of a data slice can be referenced by :code:`ds.context.slice_start` and :code:`ds.context.slice_stop`. For more details, see :ref:`Data Slice Context <data-slice-context>`.

|

**v0.4.0** July 2, 2020
    * Enhancements
        * Target values can be sampled from each group (:pr:`138`).
        * One of multiple targets can be selected (:pr:`147`).
        * Labels can be binned using infinite edges represented as string (:pr:`133`).
    * Changes
        * The label times object was refactored to improve design and structure (:pr:`135`).

.. warning::

    **Breaking Changes**
        * Loading label times from previous versions will result in an error.

|

**v0.3.0** June 1, 2020
    * Enhancements
        * Label Search for Multiple Targets (:pr:`130`)
    * Changes
        * Column renamed from :code:`cutoff_time` to :code:`time` (:pr:`139`)

**v0.2.0** April 23, 2020
    * Changes
        * Dropped Support for Python 3.5 (:pr:`128`)
        * Rename LabelTimes.name to LabelTimes.label_name (:pr:`126`)
        * Support keyword arguments in Pandas methods. (:pr:`121`)
    * Documentation Changes
        * Improved data download in Predict Next Purchase (:pr:`76`)
    * Testing Changes
        * Added tests that use Python 3.8 in CirlceCI (:pr:`128`)

.. warning::

    **Breaking Changes**
        * ``LabelTimes.name`` has been renamed to ``LabelTimes.label_name``

|

**v0.1.8** March 11, 2020
    * Fixes
        * Support for Pandas 1.0

**v0.1.7** January 31, 2020
    * Enhancements
        * Added higher-level mappings to offsets.
        * Track settings for sample transforms.
    * Fixes
        * Pinned Pandas version.
    * Testing Changes
        * Moved Featuretools to test requirements.

**v0.1.6** October 22, 2019
    * Enhancements
        * Serialization for Label Times
    * Fixes
        * Matplotlib Backend Fix
        * Sampling Label Times
    * Documentation Changes
        * Added Data Slice Generator Guide
    * Testing Changes
        * Integration Tests for Python Versions 3.6 and 3.7

**v0.1.5** September 16, 2019
    * Enhancements
        * Added Slice Generator
        * Added Seaborn Plots
        * Added Data Slice Context
        * Added Count per Group
    * Documentation Changes
        * Updated README
        * Added Example: Predict Next Purchase
        * Added Example: Predict RUL

**v0.1.4** August 7, 2019
    * Enhancements
        * Added Sample Transform
        * Improved Progress Bar
        * Improved Label Times description

**v0.1.3** July 9, 2019
    * Enhancements
        * Improved documentation
        * Added testing for Featuretools compatibility
        * Improved description of Label Times
        * Refactored search in Label Maker
        * Improved testing for Label Transforms

**v0.1.2** June 19, 2019
    * Enhancements
        * Add dynamic progress bar
        * Add label transform for binning labels
        * Improve code coverage
        * Update documentation

**v0.1.1** May 31, 2019
    * Initial Release

|
|
