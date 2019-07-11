#!/bin/sh

# Check if release was published
function published {
python - <<END
import json

with open('$GITHUB_EVENT_PATH', 'r') as file:
    event = json.load(file)
    published = event.get('action') == 'published'

print(published)
END
}

# The function that uploads to PyPI
function run {
    # Checkout specified commit
    git checkout "$TAG"

    # Remove build artifacts
    rm -rf .eggs/ rm -rf dist/ rm -rf build/

    # Create distributions
    python setup.py sdist bdist_wheel

    # Install twine, module used to upload to pypi
    pip install --user twine -q

    # Upload to pypi or testpypi
    echo "Upoading $TAG to pypitest ..."
    # python -m twine upload dist/* -r "pypitest"
}

# If release was published then upload to PyPI
if [ $(published) ]; then run; fi
