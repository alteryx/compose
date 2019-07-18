workflow "Release" {
  on = "push"
  resolves = ["PyPI"]
}

action "PyPI" {
  uses = "./release"
  secrets = ["PYPI_USERNAME", "PYPI_PASSWORD"]
  env = {
    TWINE_REPOSITORY_URL = "https://upload.pypi.org/legacy/"
    }
}