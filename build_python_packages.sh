#!/bin/bash
version=$(grep version setup.py | cut -d"=" -f2 | rev | cut -c 2- | rev | tr -d "'")
repo=testpypi
python setup.py sdist bdist_wheel
python -m twine upload --repository ${repo} "dist/diverge-${version/v/}"*
