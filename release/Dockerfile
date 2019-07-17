FROM python:3.7.3

LABEL "com.github.actions.name"="PyPI"
LABEL "com.github.actions.description"="Upload the release to PyPI when published on GitHub."
LABEL "com.github.actions.icon"="package"
LABEL "com.github.actions.color"="blue"

LABEL "repository"="https://github.com/FeatureLabs/compose-ml"
LABEL "maintainer"="support@featurelabs.com"

ADD upload.sh /upload.sh
RUN ["chmod", "+x", "upload.sh"]
ENTRYPOINT ["/upload.sh"]