.PHONY: help test run

SERVICE_NAME  ?= "ieeg-ram-data-curation"

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make run             - run the processor locally via docker-compose"
	@echo "make clean           - remove all files from the input / output directories

run:
	uv sync
	mkdir -p data/input/ram/
	mkdir -p data/output/ram/
	uv run scipts/ram/channel_metadata.py
	uv run scipts/ram/channel_metadata_check.py
	uv run scipts/ram/curate_pull_ram.py --n-patients 1

clean:
	rm -rf data/input/ram/
	rm -rf data/output/ram/
