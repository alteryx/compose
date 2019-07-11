workflow "Release" {
  on = "release"
  resolves = ["PyPI"]
}

action "PyPI" {
  uses = "./release"
  secrets = ["PYPI_USERNAME", "PYPI_PASSWORD"]
}
