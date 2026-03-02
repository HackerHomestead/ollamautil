// Package main is the entry point for the Ollama Utility CLI tool.
//
// This is a comprehensive CLI tool for managing Ollama models, written in Go.
// It provides functionality to check service health, list models, run benchmarks,
// generate performance reports, and manage (delete) models.
//
// The tool is designed to be a fast, single-binary replacement for the original
// Python implementation, offering better performance, easier deployment, and
// improved reliability.
package main

import (
	"os"

	"github.com/yourusername/ollama-util/cmd"
)

// Build-time variables set via ldflags
var (
	version = "dev"     // Set via -ldflags "-X main.version=x.y.z"
	commit  = "unknown" // Set via -ldflags "-X main.commit=abcd1234"
	date    = "unknown" // Set via -ldflags "-X main.date=2024-01-01T00:00:00Z"
)

// main is the application entry point. It delegates to the Cobra CLI framework
// for command parsing and execution. If any command returns an error, the
// application exits with status code 1.
func main() {
	// Pass version information to the CLI
	cmd.SetVersion(version, commit, date)

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
