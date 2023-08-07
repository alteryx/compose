Release Notes
-------------

Future Release
==============
    * Enhancements
    * Fixes
    * Changes
        * Remove isort, add pre-commit-config.yaml, and run on all files (:pr:`366`)
        * Specify black and ruff config arguments in pre-commit-config (:pr:`371`)
        * Update s3 bucket for docs image (:pr:`384`)
    * Documentation Changes
    * Testing Changes

    Thanks to the following people for contributing to this release:
    :user:`gsheni`:

v0.10.1 Jan 6, 2023
===================
    * Changes
        * Update create feedstock pull request workflow (:pr:`364`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`

v0.10.0 Jan 6, 2023
===================
    * Fixes
        * Update to avoid error with a categorical target with unused categories (:pr:`349`)
    * Changes
        * Transition to pure pyproject.toml for project metadata (:pr:`351`)
        * Change `target_dataframe_name` parameter name to `target_dataframe_index` (:pr:`353`)
        * Delete MANIFEST.in and .coveragerc from root directory (:pr:`359`)
    * Documentation Changes
        * Temporarily restrict scikit-learn version to ``<1.2.0`` in dev requirements to allow docs to build (:pr:`361`)
    * Testing Changes
        * Add create feedstock PR workflow (:pr:`346`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`thehomebrewnerd`

Breaking Changes
++++++++++++++++
* The parameter ``target_dataframe_name`` has been changed to ``target_dataframe_index`` in ``LabelMaker``.

v0.9.1 Nov 2, 2022
==================
    * Changes
        * Explicitly set series dtype for ``LabelTimes.target_types`` (:pr:`337`)
    * Documentation Changes
        * Fix docs build and clean up release notes (:pr:`336`)

    Thanks to the following people for contributing to this release:
    :user:`thehomebrewnerd`

v0.9.0 May 12, 2022
===================
    .. warning::
        Compose will no longer support Python 3.7.

    * Changes
        * Update ipython to 7.31.1 (:pr:`286`)
        * Transition to pyproject.toml and setup.cfg (:pr:`310`, :pr:`313`)
        * Add support for python 3.10 (:pr:`318`)
        * Fix Makefile output filepath (:pr:`320`)
    * Documentation Changes
        * Update README.md with Alteryx link (:pr:`289`, :pr:`290`, :pr:`314`)
        * Add in-line tabs and copy-paste functionality to docs (:pr:`293`)
        * Update nbconvert to version 6.4.5 to fix docs build issue (:pr:`305`)
        * Update slack invite link to new (:pr:`316`)
        * Update ``release.md`` with correct process (:pr:`324`)
    * Testing Changes
        * Add woodwork to ``test-requirements.txt`` (:pr:`296`)
        * Upgrade black version to 22.3.0 to fix linting issue (:pr:`309`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`mingdavidqi`, :user:`thehomebrewnerd`

v0.8.0 Jan 20, 2022
===================
    * Enhancements
        * Add issue templates for bugs, feature requests and documentation improvements (:pr:`271`)
    * Changes
        * Update pip to 21.3.1 for test requirements (:pr:`265`)
        * Restrict to Python 3.7 to 3.9 (:pr:`265`)
        * Use black and remove autopep8 for linting (:pr:`265`)
        * Update minimum dependency checker with the correct reviewers (:pr:`267`)
        * Rename ``LabelMaker.target_entity`` to ``LabelMaker.target_dataframe_name`` (:pr:`276`)
    * Documentation Changes
        * Update install instructions to specify correct python versions (:pr:`265`)
        * Update example notebooks to use latest Featuretools and EvalML APIs (:pr:`275`)
    * Testing Changes
        * Add unit test for dropping empty data slices (:pr:`280`)
        * Add auto approve workflow for dependency updates (:pr:`281`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`, :user:`thehomebrewnerd`

    .. warning::

        **Breaking Changes**
            * The ``target_entity`` attribute of ``LabelMaker`` has been renamed to ``target_dataframe_name``.

v0.7.0 Nov 2, 2021
==================
    * Enhancements
        * Add ``maximum_data`` parameter to control when a search should stop (:pr:`216`)
        * Add optional automatic update checker (:pr:`223`, :pr:`229`, :pr:`232`)
        * Varying first cutoff time for each target group (:pr:`258`)
    * Documentation Changes
        * Update doc tutorials to the latest API changes (:pr:`227`)
        * Pin documentation requirements to avoid warnings and breaking changes (:pr:`244`)
    * Testing Changes
        * Check if release notes were updated (:pr:`217`)
        * Add minimum dependency checker to generate minimum requirement files (:pr:`218`)
        * Add CI workflow for unit tests with minimum dependencies (:pr:`220`)
        * Create separate worksflows for each CI job (:pr:`220`)
        * Pass token to authorize uploading of codecov reports (:pr:`226`)
        * Update minimum unit tests to run on all pull requests (:pr:`230`)
        * Add workflow to check latest dependencies (:pr:`233`)
        * Update reviewers for minimum and latest dependency checkers (:pr:`257`)

    Thanks to the following people for contributing to this release:
    :user:`gsheni`, :user:`jeff-hernandez`

v0.6.0 Feb 11, 2021
===================
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

    Thanks to the following people for contributing to this release:
    :user:`flowersw`, :user:`jeff-hernandez`, :user:`rwedge`

v0.5.1 Sep 22, 2020
===================
    * Documentation Changes
        * Update F1 Macro in Turbofan Degradation Tutorial (:pr:`180`).
        * Apply Pandas Docs Theme (:pr:`172`).
        * Add Chicago Bike Tutorial (:pr:`157`).
    * Testing Changes
        * Test Doc Builds (:pr:`165`)

v0.5.0 Aug 28, 2020
===================
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

v0.4.0 Jul 2, 2020
==================
    * Enhancements
        * Target values can be sampled from each group (:pr:`138`).
        * One of multiple targets can be selected (:pr:`147`).
        * Labels can be binned using infinite edges represented as string (:pr:`133`).
    * Changes
        * The label times object was refactored to improve design and structure (:pr:`135`).

    .. warning::

        **Breaking Changes**
            * Loading label times from previous versions will result in an error.

v0.3.0 Jun 1, 2020
==================
    * Enhancements
        * Label Search for Multiple Targets (:pr:`130`)
    * Changes
        * Column renamed from :code:`cutoff_time` to :code:`time` (:pr:`139`)

v0.2.0 Apr 23, 2020
===================
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

v0.1.8 Mar 11, 2020
===================
    * Fixes
        * Support for Pandas 1.0

v0.1.7 Jan 31, 2020
===================
    * Enhancements
        * Added higher-level mappings to offsets.
        * Track settings for sample transforms.
    * Fixes
        * Pinned Pandas version.
    * Testing Changes
        * Moved Featuretools to test requirements.

v0.1.6 Oct 22, 2019
===================
    * Enhancements
        * Serialization for Label Times
    * Fixes
        * Matplotlib Backend Fix
        * Sampling Label Times
    * Documentation Changes
        * Added Data Slice Generator Guide
    * Testing Changes
        * Integration Tests for Python Versions 3.6 and 3.7

v0.1.5 Sep 16, 2019
===================
    * Enhancements
        * Added Slice Generator
        * Added Seaborn Plots
        * Added Data Slice Context
        * Added Count per Group
    * Documentation Changes
        * Updated README
        * Added Example: Predict Next Purchase
        * Added Example: Predict RUL

v0.1.4 Aug 7, 2019
==================
    * Enhancements
        * Added Sample Transform
        * Improved Progress Bar
        * Improved Label Times description

v0.1.3 Jul 9, 2019
==================
    * Enhancements
        * Improved documentation
        * Added testing for Featuretools compatibility
        * Improved description of Label Times
        * Refactored search in Label Maker
        * Improved testing for Label Transforms

v0.1.2 Jun 19, 2019
===================
    * Enhancements
        * Add dynamic progress bar
        * Add label transform for binning labels
        * Improve code coverage
        * Update documentation

v0.1.1 May 31, 2019
===================
    * Initial Release
