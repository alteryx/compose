#!/bin/bash

# Clone featuretools_update_checker repo
git clone https://github.com/FeatureLabs/featuretools_update_checker.git /home/circleci/featuretools_update_checker
# Checkout specified commit
cd /home/circleci/featuretools_update_checker
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