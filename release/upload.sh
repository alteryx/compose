#!/bin/sh

upload_to_pypi () {
    # Get tag from environment
    tag=$(basename $GITHUB_REF)

    # Checkout specified tag
    git checkout tags/$tag

    # Remove build artifacts
    rm -rf .eggs/ rm -rf dist/ rm -rf build/

    # Create distributions
    python setup.py sdist bdist_wheel -q

    # Install twine, module used to upload to pypi
    pip install --user twine -q

    # Upload to pypi or testpypi
    echo "Uploading $tag to $TWINE_REPOSITORY_URL ..."
    TWINE_USERNAME=$PYPI_USERNAME
    TWINE_PASSWORD=$PYPI_PASSWORD
    python -m twine upload dist/*
}

# Get action that triggered release event
action=$(python -c "
import json

with open('$GITHUB_EVENT_PATH', 'r') as file:
    event = json.load(file)
    action = event.get('action')

print(action)
")

echo "Release was $action on GitHub ..."

# If release was published on GitHub then upload to PyPI
if [ $action = "published" ]; then upload_to_pypi; fi
