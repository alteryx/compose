#!/bin/sh

# Clone repository
git clone "https://github.com/${1}.git" "$HOME/project"

# Checkout specified commit
cd "$HOME/project"
git checkout "$TAG"

# Remove build artifacts
rm -rf .eggs/ rm -rf dist/ rm -rf build/

# Create distributions
python setup.py sdist bdist_wheel

# Install twine, module used to upload to pypi
pip install --user twine -q

# Upload to pypi or testpypi
echo "Upoading $TAG to pypitest ${2} ..."
# python -m twine upload dist/* -r "pypitest"
