# English version uses pdflatex (pdflatex -> bibtex -> pdflatex -> pdflatex)
# The Chinese version main_zh.tex needs xeCJK, so it is switched to xelatex automatically below
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$pdf_mode = 5 if grep { /main_zh(\.tex)?$/ } @ARGV;
$bibtex_use = 2;
