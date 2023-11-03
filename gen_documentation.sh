#!/bin/bash

./build_python_packages.sh clean

if [ $(uname -o) = "Darwin" ]
then
    nd_exe=naturaldocs
else
    nd_exe=naturaldocs2
fi

rm -rf public/{classes/,files/,index.html,menu/,search/,styles/}
rm -rf nd_conf/Working\ Data
$nd_exe nd_conf/Project.txt
