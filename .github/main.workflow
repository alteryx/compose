workflow "PyPI Release" {
  on = "release"
  resolves = ["release"]
}

action "release" {
  uses = "./release@github-actions"
  args = "$GITHUB_REF"
}
