workflow "Publish" {
  on = "push"
  resolves = ["PyPI Release"]
}

action "PyPI Release" {
  uses = "./release"
  env = {
    TAG  = "v0.1.3"
  }
}
