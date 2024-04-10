#!/bin/bash

mypath="$(pwd)/$(dirname "$0")"

cd "$mypath"
rm -rf ../public/*.html ../public/*.js ../public/*.inv ../public/_*
cp -r _build/html/* ../public/
