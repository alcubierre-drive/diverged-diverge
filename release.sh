#!/bin/bash

TAG_VERSION="$(git tag --list "v*.*" | tail -n1)"
mkdir -p public/releases/${TAG_VERSION}/cpu
cp divERGe libdivERGe.so public/releases/${TAG_VERSION}/cpu
cp divERGe.tar.gz divERGe_src.tar.gz public/releases/${TAG_VERSION}

