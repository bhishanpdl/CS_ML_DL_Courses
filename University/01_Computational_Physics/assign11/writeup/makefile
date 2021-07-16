# LaTeX Makefile for dvi, ps, and pdf file creation.
# Usage:
# make          # make dvi, ps, and pdf
# make dvi      # make dvi
# make ps       # make ps (and dvi)
# make pdf      # make pdf
#

MAIN        = hw11
SOURCES	    = $(wildcard ./*.tex)
EPSFIGURES	= $(patsubst %.fig,%.eps,$(wildcard ./figures/*.fig))
PDFFIGURES	= $(patsubst %.fig,%.pdf,$(wildcard ./figures/*.fig))


# set pdfviewer for linux and unix machines
####################################################
UNAME_S := $(shell uname -s)

$(info $$UNAME_S == $(UNAME_S))

ifeq ($(UNAME_S),Linux)
  PDFVIEWER := xdg-open
else ifeq ($(UNAME_S),Darwin)
  PDFVIEWER := open
else
  $(error unsupported system: $(UNAME_S))
endif

$(info $$PDFVIEWER == $(PDFVIEWER))
####################################################

# open the pdf and delete junk files
default: all
	$(PDFVIEWER) $(MAIN).pdf         
	rm -f *.aux *.lof *.log *.out *.ps 
	rm -f *.toc *.dvi 
	rm -f *.synctex.gz *.bak *~
	rm -f *-eps-converted*
	rm -f figures/*-eps-converted* 

all: dvi ps pdf

dvi: ${MAIN}.dvi
pdf: ${MAIN}.pdf
ps: ${MAIN}.ps

${MAIN}.dvi : ${SOURCES} ${EPSFIGURES}
	latex ${MAIN}
	@while ( grep "Rerun to get cross-references"	\
	${MAIN}.log > /dev/null ); do		\
	echo '** Re-running LaTeX **';		\
	latex ${MAIN};				\
	done

${MAIN}.pdf : ${SOURCES} ${PDFFIGURES}
	pdflatex ${MAIN}
	@while ( grep "Rerun to get cross-references" 	\
	${MAIN}.log > /dev/null ); do		\
	echo '** Re-running LaTeX **';		\
	pdflatex ${MAIN};				\
	done

${MAIN}.ps : ${MAIN}.dvi
        # running dvips
	dvips ${MAIN}.dvi -o ${MAIN}.ps
clean:
	rm -f *.aux *.lof *.log *.out *.ps *.toc *.dvi 
	rm -f *.synctex.gz *.bak *~
	rm -f *-eps-converted* 
#
# (re)Make .eps is .fig if newer
#
%.eps : %.fig
	#Creates .eps file
	fig2dev -L pstex $*.fig > $*.eps
	#Creates .tex file
	fig2dev -L pstex_t -p $* $*.fig > $*.tex
#
# (re)Make .pdf if .eps is newer
#
%.pdf : %.eps
	#Creates .pdf files from .eps files
	epstopdf $*.eps > $*.pdf

