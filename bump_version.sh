#!/bin/bash
TAG_VERSION="$(git tag --list | tac | head -n-1 | sort -V | tail -n1)"

echo $TAG_VERSION

FILE=README.md
sed -i 's|v[0-9]*\.[0-9]*|'${TAG_VERSION}'|g' "$FILE"
