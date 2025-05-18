# Makefile for compiling Protocol Buffer files

.PHONY: all clean install-deps compile-protos

all: install-deps compile-protos

# Install dependencies
install-deps:
	pip install grpcio grpcio-tools

# Compile Protocol Buffer files
compile-protos:
	python compile_protos.py

# Clean generated files
clean:
	rm -rf ../common_lib/grpc