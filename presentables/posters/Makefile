# You can change the pdf viewer to your preferred
# one by commenting every line beginning by
# `PDFVIEWER' except the one with your pdf viewer
PDFVIEWER=evince # GNOME
#PDFVIEWER=okular # KDE
#PDFVIEWER=xpdf # lightweight
#PDFVIEWER=xdg-open # Default pdf viewer - GNU/Linux
#PDFVIEWER=open # Default pdf viewer - Mac OS
MAIN_NAME=Prieels-Quentin-poster
PDF_NAME=$(MAIN_NAME).pdf

# You want latexmk to *always* run, because make does not have all the info.
.PHONY: $(PDF_NAME) clean show release

# If you want the pdf to be opened by your preferred pdf viewer
# after `$ make', comment the following line and uncomment the
# line after
default: all
#default: show

all: $(PDF_NAME)

# MAIN LATEXMK RULE

# -pdf tells latexmk to generate PDF directly (instead of DVI).
# -pdflatex="" tells latexmk to call a specific backend with specific options.
# -use-make tells latexmk to call make for generating missing files.

# -interactive=nonstopmode keeps the pdflatex backend from stopping at a
# missing file reference and interactively asking you for an alternative.

# Other depences are gessed automatically by latexmk
# see http://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project
$(PDF_NAME): $(MAIN_NAME).tex
	latexmk -pdf -xelatex -use-make $(MAIN_NAME).tex 

clean:
	latexmk -CA
	$(RM) $(MAIN_NAME).bbl

show: $(PDF_NAME)
	$(PDFVIEWER) $(PDF_NAME) 2> /dev/null &

publish: $(PDF_NAME)
	cp $(PDF_NAME) ~/Dropbox/Research/$(NAME)

release:
	if [ -a config.yml ] ; \
	then \
		python3 ~/git/smartcp/smartcp.py --google-drive -vvvvv config.yml ; \
	else \
		cp $(MAIN_NAME).pdf ~/gdrive/Research/$(TYPE)/$(NAME).pdf; \
		drive push ~/gdrive/Research/$(TYPE)/$(NAME).pdf; \
	fi;
ifdef PDFNAME
	if ! [ -a config.yml ] ; \
	then \
		cp $(MAIN_NAME).pdf "/home/blegat/gdrive/Research benoit et raph/Final versions/$(PDFNAME).pdf"; \
		drive push "/home/blegat/gdrive/Research benoit et raph/Final versions/$(PDFNAME).pdf"; \
	fi;
endif
