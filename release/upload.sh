#!/bin/bash

# Clone compose-ml repo
git clone https://github.com/FeatureLabs/compose-ml.git /home/circleci/compose-ml
# Checkout specified commit
cd /home/circleci/compose-ml
git checkout "${1}"
# Remove build artifacts
rm -rf .eggs/ rm -rf dist/ rm -rf build/
# Create distributions
python setup.py sdist bdist_wheel
# Install twine, module used to upload to pypi
pip install --user twine
# Upload to pypi or testpypi
echo "Upoading to ${2:-pypi} . . ."
python -m twine upload dist/* -r "${2:-pypi}"