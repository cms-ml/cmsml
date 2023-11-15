<!-- marker-before-logo -->

<p align="center">
  <a href="https://github.com/cms-ml/cmsml">
    <img src="https://raw.githubusercontent.com/cms-ml/cmsml/master/logo.png" />
  </a>
</p>

<!-- marker-after-logo -->

<!-- marker-before-badges -->

<p align="center">
  <a href="https://github.com/cms-ml/cmsml/actions?query=workflow%3A%22Lint+and+test%22">
    <img alt="Lint and test" src="https://github.com/cms-ml/cmsml/workflows/Lint%20and%20test/badge.svg" />
  </a>
  <a href="https://github.com/cms-ml/cmsml/actions?query=workflow%3A%22Deploy+images%22">
    <img alt="Deploy images" src="https://github.com/cms-ml/cmsml/workflows/Deploy%20images/badge.svg" />
  </a>
  <a href="http://cmsml.readthedocs.io">
    <img alt="Documentation status" src="https://readthedocs.org/projects/cmsml/badge/?version=latest" />
  </a>
  <img alt="Python version" src="https://img.shields.io/badge/Python-%E2%89%A53.7-blue" />
  <a href="https://pypi.python.org/pypi/cmsml">
    <img alt="Package version" src="https://img.shields.io/pypi/v/cmsml.svg?style=flat" />
  </a>
  <a href="https://github.com/cms-ml/cmsml/blob/master/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/cms-ml/cmsml.svg" />
  </a>
</p>

<!-- marker-after-badges -->

<!-- marker-before-header -->

## CMS Machine Learning Group Python package.

The documentation of this Python package is hosted on [readthedocs](http://cmsml.readthedocs.io).

**However**, note that this documentation only covers the API and technical aspects of the package itself.
Usage examples and further techniques for working with machine learning tools in CMS, alongside a collection of useful guidelines can be found in the [general CMS ML group documentation](https://cms-ml.github.io/documentation).

Click [here](https://github.com/cms-ml/cmsml/issues/new?labels=suggestion&template=feature-suggestion.md&) to submit a feature suggestion!


<!-- marker-after-header -->

<!-- marker-before-body -->

<!-- marker-before-docker -->

## Docker images

To use the cmsml package via docker, checkout our [DockerHub](https://hub.docker.com/repository/docker/cmsml/cmsml) which contains tags for several Python versions.

<!-- marker-after-docker -->

<!-- marker-before-testing -->

## Testing

The tests can be triggered with

```shell
pytest -n auto tests
```

and in general, they should be run for Python 3.7 to 3.11.
To run tests in a docker container, do

```shell
# run the tests
./tests/docker.sh cmsml/cmsml

# or interactively by adding a flag "i" to the command
./tests/docker.sh cmsml/cmsml i
> python -m unittest tests
```

In addition, before pushing to the repository, [PEP 8](https://www.python.org/dev/peps/pep-0008) compatibility should be checked with [flake8](https://pypi.org/project/flake8) via

```shell
./tests/lint.sh
```

or using the docker container

```shell
# run the tests
./tests/docker.sh cmsml/cmsml tests/lint.sh
```

<!-- marker-after-testing -->

<!-- marker-before-development -->

- Source hosted at [GitHub](https://github.com/cms-ml/cmsml)
- Report issues, questions, feature requests on [GitHub Issues](https://github.com/cms-ml/cmsml/issues)

<!-- marker-after-development -->

<!-- marker-after-body -->
