#!/bin/bash

if [ $1 = "clean" ]
then
    rm -rf build diverge_flow.egg-info dist
    exit 0
fi

version=$(grep version setup.py | cut -d"=" -f2 | rev | cut -c 2- | rev | tr -d "'")
repo=pypi
python setup.py sdist bdist_wheel
python -m twine upload --repository ${repo} "dist/diverge-flow-${version/v/}"*
