workflow "Publish Release" {
  on = "push"
  resolves = ["Upload to PyPI"]
}

action "Upload to PyPI" {
  uses = "./release"
  env = {
    TAG  = "v0.1.3"
  }
  args = ["$GITHUB_REPOSITORY", "$GITHUB_REF"]
}
