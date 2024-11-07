# Description: Makefile for compiling LaTeX documents
# Usage: make <project-name>

# Variables
LATEX=pdflatex
PROJECT=$(firstword $(MAKECMDGOALS))
SOURCE=$(PROJECT).tex
OUTPUT=$(PROJECT).pdf

# Default target
all: $(PROJECT)

# Compile the LaTeX document
$(PROJECT): $(PROJECT)/$(SOURCE)
	@echo "Compiling $(PROJECT)..."
	@cd $(PROJECT) && $(LATEX) --shell-escape -jobname=$(PROJECT) $(SOURCE)
	@echo "Cleaning up auxiliary files..."
	@cd $(PROJECT) && rm -f $(PROJECT).{aux,log,out,toc,synctex.gz,nav,snm,bcf,xml} >> /dev/null
	@echo "Done!"

# Phony targets
.PHONY: all $(PROJECT)