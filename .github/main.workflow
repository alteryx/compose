workflow "PyPI Release" {
  on = "push"
  resolves = ["release"]
}

action "release" {
  uses = "./release@github-actions"
  args = "$GITHUB_REF"
}
