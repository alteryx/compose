FROM python:3.7.3

LABEL "com.github.actions.name"="Upload to PyPI"
LABEL "com.github.actions.description"="Upload to PyPI on release."
LABEL "com.github.actions.icon"="package"
LABEL "com.github.actions.color"="blue"

LABEL "repository"="https://github.com/FeatureLabs/compose-ml"
LABEL "maintainer"="support@featurelabs.com"

ADD upload.sh /upload.sh
RUN ["chmod", "+x", "upload.sh"]
ENTRYPOINT ["/upload.sh"]