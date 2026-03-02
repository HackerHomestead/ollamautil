# Makefile for ollama-util

APP_NAME := ollama-util
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
DATE := $(shell date -u '+%Y-%m-%dT%H:%M:%SZ')
LDFLAGS := -ldflags "-X main.version=$(VERSION) -X main.commit=$(COMMIT) -X main.date=$(DATE)"

# Default target
.PHONY: build
build:
	go build $(LDFLAGS) -o $(APP_NAME) .

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(APP_NAME)
	rm -rf dist/

# Run tests
.PHONY: test
test:
	go test -v ./...

# Format code
.PHONY: fmt
fmt:
	go fmt ./...

# Lint code
.PHONY: lint
lint:
	golangci-lint run

# Cross-platform builds
.PHONY: build-all
build-all: clean
	mkdir -p dist
	
	# Linux AMD64
	GOOS=linux GOARCH=amd64 go build $(LDFLAGS) -o dist/$(APP_NAME)-linux-amd64 .
	
	# Linux ARM64
	GOOS=linux GOARCH=arm64 go build $(LDFLAGS) -o dist/$(APP_NAME)-linux-arm64 .
	
	# macOS AMD64
	GOOS=darwin GOARCH=amd64 go build $(LDFLAGS) -o dist/$(APP_NAME)-darwin-amd64 .
	
	# macOS ARM64 (Apple Silicon)
	GOOS=darwin GOARCH=arm64 go build $(LDFLAGS) -o dist/$(APP_NAME)-darwin-arm64 .
	
	# Windows AMD64
	GOOS=windows GOARCH=amd64 go build $(LDFLAGS) -o dist/$(APP_NAME)-windows-amd64.exe .
	
	# Windows ARM64
	GOOS=windows GOARCH=arm64 go build $(LDFLAGS) -o dist/$(APP_NAME)-windows-arm64.exe .

# Create release archives
.PHONY: release
release: build-all
	cd dist && \
	tar -czf $(APP_NAME)-linux-amd64.tar.gz $(APP_NAME)-linux-amd64 && \
	tar -czf $(APP_NAME)-linux-arm64.tar.gz $(APP_NAME)-linux-arm64 && \
	tar -czf $(APP_NAME)-darwin-amd64.tar.gz $(APP_NAME)-darwin-amd64 && \
	tar -czf $(APP_NAME)-darwin-arm64.tar.gz $(APP_NAME)-darwin-arm64 && \
	zip $(APP_NAME)-windows-amd64.zip $(APP_NAME)-windows-amd64.exe && \
	zip $(APP_NAME)-windows-arm64.zip $(APP_NAME)-windows-arm64.exe

# Install locally
.PHONY: install
install: build
	cp $(APP_NAME) /usr/local/bin/

# Development server with hot reload
.PHONY: dev
dev:
	go run . check

# Benchmark the tool itself
.PHONY: bench
bench:
	go test -bench=. ./...

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build      - Build the application"
	@echo "  build-all  - Build for all supported platforms"
	@echo "  test       - Run tests"
	@echo "  fmt        - Format code"
	@echo "  lint       - Run linter"
	@echo "  clean      - Clean build artifacts"
	@echo "  install    - Install binary to /usr/local/bin"
	@echo "  release    - Create release archives"
	@echo "  dev        - Quick development test"
	@echo "  help       - Show this help"