workflow "Release" {
  on = "push"
  resolves = ["PyPI"]
}

action "PyPI" {
  uses = "./release"
  env = {
    TAG  = "v0.1.3"
  }
}
