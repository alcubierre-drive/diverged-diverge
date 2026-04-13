#!/bin/bash

# XXX: Do not invoke separately. Use `make dist`.

TAG_VERSION="$(git tag --list | tac | head -n-1 | sort -V | tail -n1)"

echo $TAG_VERSION

mkdir -p public/releases/${TAG_VERSION}/cpu
cp divERGe libdivERGe.so public/releases/${TAG_VERSION}/cpu
cp divERGe.tar.gz divERGe_src.tar.gz public/releases/${TAG_VERSION}

