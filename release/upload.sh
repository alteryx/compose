#!/bin/sh

# Get action that triggered release event
release=$(python -c "
import json

with open('$GITHUB_EVENT_PATH', 'r') as file:
    event = json.load(file)
    release = event.get('action')

print(release)
")

upload_to_pypi () {
    # Checkout specified commit
    git checkout tags/$GITHUB_REF

    # Remove build artifacts
    rm -rf .eggs/ rm -rf dist/ rm -rf build/

    # Create distributions
    python setup.py sdist bdist_wheel

    # Install twine, module used to upload to pypi
    pip install --user twine -q

    # Upload to pypi or testpypi
    echo "Uploading $GITHUB_REF to $TWINE_REPOSITORY_URL ..."
    TWINE_USERNAME=$PYPI_USERNAME
    TWINE_PASSWORD=$PYPI_PASSWORD
    twine upload dist/*
}

# If release was published on GitHub then upload to PyPI
if [ $release = "published" ]; then upload_to_pypi; fi
