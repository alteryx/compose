# Install

Compose is available for Python 3.8, 3.9, 3.10, and 3.11. It can be installed from [PyPI](https://pypi.org/project/composeml/), [conda-forge](https://anaconda.org/conda-forge/composeml), or from [source](https://github.com/alteryx/compose).

## pip

To install Compose, run the following command:

````{tab} PyPI
```console
$ python -m pip install composeml
```
````

````{tab} Conda
```console
$ conda install -c conda-forge composeml
```
````

````{tab} Source
```console
git clone https://github.com/alteryx/compose.git
cd compose
python -m pip install .
```
````

## Docker
It is also possible to run Compose inside a Docker container.

You can do so by installing it as a package inside a container (following the normal install guide) or
creating a new image with Compose pre-installed, using the following commands in your Dockerfile:

```bash
FROM python:3.8-slim-buster
RUN apt-get update && apt-get -y update
RUN apt-get install -y build-essential python3-pip python3-dev
RUN pip -q install pip --upgrade
RUN pip install composeml
```

## Add-ons

* Update checker: Receive automatic notifications of new Compose releases

````{tab} PyPI
```console
$ python -m pip install composeml[update_checker]
```
````

````{tab} Conda
```console
$ conda install -c conda-forge alteryx-open-src-update-checker
```
````
