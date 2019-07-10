#!/bin/sh
# Clone compose-ml repo
git clone https://github.com/FeatureLabs/compose-ml.git $HOME/compose-ml
# Checkout specified commit
cd $HOME/compose-ml
git checkout "$TAG"
# Remove build artifacts
rm -rf .eggs/ rm -rf dist/ rm -rf build/
# Create distributions
python setup.py sdist bdist_wheel
# Install twine, module used to upload to pypi
pip install --user twine
# Upload to pypi or testpypi
echo "Upoading $TAG to pypitest . . ."
# python -m twine upload dist/* -r "pypitest"