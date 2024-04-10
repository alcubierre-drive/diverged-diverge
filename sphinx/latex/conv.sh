#!/bin/bash
for f in *.tex 
do
    latexmk $f
    pdf_file=$(echo $f | sed 's/tex/pdf/g')
    png_file=$(echo $f | sed 's/tex/png/g')
    convert \
        -density 150 $pdf_file \
        -background white -alpha remove -alpha off ../img/$png_file
done
