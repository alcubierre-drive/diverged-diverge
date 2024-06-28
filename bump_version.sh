#!/bin/bash
TAG_VERSION="$(git tag --list "v*.*" | tail -n1)"

FILE=README.md
sed -i 's|v[0-9]*\.[0-9]*|'${TAG_VERSION}'|g' "$FILE"
