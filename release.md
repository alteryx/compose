# Release Process
## Create composeml release on github
#### Create release branch
1. Branch off of master and name the branch the release version number (e.g. v0.1.2)
2. Bump verison number in `setup.py`, and `composeml/__init__.py`.

#### Create Release PR
A release PR should have the version number as the title and the changelog updates as the PR body text. The contributors line is not necessary.

#### Create Github Release
After the release pull request has been merged into the master branch, it is time draft the github release.
* The target should be the master branch
* The tag should be the version number with a v prefix (e.g. v0.1.2)
* Release title is the same as the tag
* Release description should be the full changelog updates for the release, including the line thanking contributors.

## Release on PyPI
1. Update circleci's python3 image
    ```bash
    docker pull circleci/python:3
    ```
2. Run upload script
    * Replace `/absolute/path/to/upload.sh` with the actual path
    * Replace the "release_tag" part of `tags/release_tag` with the actual tag
    ```bash
    docker run \
        --rm \
        -it \
        -v /absolute/path/to/upload.sh:/home/circleci/upload.sh \
        circleci/python:3 \
        /bin/bash -c "bash /home/circleci/upload.sh tags/release_tag"
    ```
