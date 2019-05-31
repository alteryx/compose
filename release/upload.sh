#!/bin/bash

# Clone composeml repo
git clone https://github.com/FeatureLabs/composeml.git /home/circleci/composeml
# Checkout specified commit
cd /home/circleci/composeml
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