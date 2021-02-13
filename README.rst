.. figure:: https://raw.githubusercontent.com/cms-ml/cmsml/master/logo.png
   :target: https://github.com/cms-ml/cmsml
   :alt: cmsml logo

.. marker-after-logo


.. image:: https://github.com/cms-ml/cmsml/workflows/Lint%20and%20test/badge.svg
   :target: https://github.com/cms-ml/cmsml/actions?query=workflow%3A%22Lint+and+test%22
   :alt: Lint and test

.. image:: https://github.com/cms-ml/cmsml/workflows/Deploy%20images/badge.svg
   :target: https://github.com/cms-ml/cmsml/actions?query=workflow%3A%22Deploy+images%22
   :alt: Deploy images

.. image:: https://readthedocs.org/projects/cmsml/badge/?version=latest
   :target: http://cmsml.readthedocs.io/en/latest
   :alt: Documentation status

.. image:: https://img.shields.io/pypi/v/cmsml.svg?style=flat
   :target: https://pypi.python.org/pypi/cmsml
   :alt: Package version

.. image:: https://img.shields.io/github/license/cms-ml/cmsml.svg
   :target: https://github.com/cms-ml/cmsml/blob/master/LICENSE
   :alt: License

.. marker-after-badges


CMS Machine Learning Group Python package
=========================================

.. note::
   This project is under development. Click `here <https://github.com/cms-ml/cmsml/issues/new?labels=suggestion&template=feature-suggestion.md&>`__ to submit a feature suggestion!

.. marker-after-header


Testing
-------

The tests can be triggered with

.. code-block:: shell

   python -m unittest tests

and in general, they should be run for Python 2.7, 3.7 and 3.8. To run tests in a docker container, do

.. code-block:: shell

   # run the tests
   ./tests/docker.sh cmsml/cmsml

   # or interactively by adding a flag "i" to the command
   ./tests/docker.sh cmsml/cmsml i
   > python -m unittest tests

In addition, before pushing to the repository, `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`__ compatibility should be checked with `flake8 <https://pypi.org/project/flake8/>`__

.. code-block:: shell

   ./tests/lint.sh

or via using the docker container

.. code-block:: shell

   # run the tests
   ./tests/docker.sh cmsml/cmsml tests/lint.sh


Development
-----------

- Source hosted at `GitHub <https://github.com/cms-ml/cmsml>`__
- Report issues, questions, feature requests on `GitHub Issues <https://github.com/cms-ml/cmsml/issues>`__

.. marker-after-content
