workspace(name = "yolox")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Python
http_archive(
    name = "rules_python",
    sha256 = "5868e73107a8e85d8f323806e60cad7283f34b32163ea6ff1020cf27abef6036",
    strip_prefix = "rules_python-0.25.0",
    urls = [
        "https://github.com/bazelbuild/rules_python/releases/download/0.25.0/rules_python-0.25.0.tar.gz"
    ],
)

# Keep all rule loadings at the bottom, otherwise the python rules don't work.
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

local_repository(
    name = "ssml_datautil_py",
    path = "../ssml_datautil_py",
)
