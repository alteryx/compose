workflow "PyPI Release" {
  on = "release"
  resolves = ["release"]
}

action "release" {
  uses = "./release"
  args = "$GITHUB_REF"
}
