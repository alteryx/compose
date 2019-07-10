#!/bin/sh

published=$(python -c "
import json
file = open('$GITHUB_EVENT_PATH', 'r')
event = json.load(file)
published = event.get('action') == published
print(published)")

echo $published

# # Checkout specified commit
# git checkout "$TAG"

# # Remove build artifacts
# rm -rf .eggs/ rm -rf dist/ rm -rf build/

# # Create distributions
# python setup.py sdist bdist_wheel

# # Install twine, module used to upload to pypi
# pip install --user twine -q

# # Upload to pypi or testpypi
# echo "Upoading $TAG to pypitest ..."
# # python -m twine upload dist/* -r "pypitest"
