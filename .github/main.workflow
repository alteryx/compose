workflow "PyPI Release" {
  on = "push"
  resolves = ["release"]
}

action "release" {
  uses = "./release"
  args = "\"test\""
}
