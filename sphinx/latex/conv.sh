#!/bin/bash
for f in *.tex 
do
    latexmk $f
    pdf_file=$(echo $f | sed 's/tex/pdf/g')
    png_file=$(echo $f | sed 's/tex/png/g')
    magick \
        -density 300 $pdf_file \
        -background white -alpha remove -alpha off ../img/$png_file
done
