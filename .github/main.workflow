workflow "Release" {
  on = "release"
  resolves = ["PyPI"]
}

action "PyPI" {
  uses = "FeatureLabs/gh-action-pypi-upload@master"
  secrets = ["PYPI_USERNAME", "PYPI_PASSWORD"]
  env = {
    TWINE_REPOSITORY_URL = "https://test.pypi.org/legacy/"
    }
}