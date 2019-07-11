#!/bin/sh

# Check if release was published on GitHub
published_on_github=$(python -c "
import json

with open('$GITHUB_EVENT_PATH', 'r') as file:
    event = json.load(file)
    published = event.get('action') == 'published'

print(published)
")

upload_to_pypi () {
    # Checkout specified commit
    git checkout "$GITHUB_REF"

    # Remove build artifacts
    rm -rf .eggs/ rm -rf dist/ rm -rf build/

    # Create distributions
    python setup.py sdist bdist_wheel

    # Install twine, module used to upload to pypi
    pip install --user twine -q

    # Upload to pypi or testpypi
    echo "Uploading $GITHUB_REF to pypitest ..."
    python -m twine upload dist/* -r "pypitest" \
    --username $PYPI_USERNAME --password $PYPI_PASSWORD
}

# If release was published on GitHub then upload to PyPI
if [ $published_on_github = "True" ]; then upload_to_pypi; fi
