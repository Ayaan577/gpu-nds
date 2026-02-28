@echo off
set PATH=C:\Users\Ayaan\AppData\Local\Programs\MiKTeX\miktex\bin\x64;%PATH%
cd /d c:\Users\Ayaan\Downloads\SM_FN\gpu_nds\paper
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
bibtex main
pdflatex -interaction=nonstopmode main.tex >nul 2>&1
pdflatex -interaction=nonstopmode main.tex 2>&1 | findstr "Output written" "undefined"
