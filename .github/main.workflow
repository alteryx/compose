workflow "PyPI Release" {
  on = "push"
  resolves = ["release"]
}

action "release" {
  uses = "./release"
  args = "$GITHUB_REF"
}
