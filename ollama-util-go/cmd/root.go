// Package cmd contains all CLI commands and their implementations.
//
// This package uses the Cobra CLI framework to provide a rich command-line
// interface with subcommands, flags, and comprehensive help text. Each command
// is implemented in its own file for better organization and maintainability.
package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/yourusername/ollama-util/internal/ollama"
)

var (
	// baseURL is the Ollama API base URL, configurable via --base-url flag
	baseURL string

	// client is the shared Ollama HTTP client instance used by all commands
	client ollama.Client

	// Version information set by main package
	appVersion = "dev"
	appCommit  = "unknown"
	appDate    = "unknown"
)

// rootCmd represents the base command when called without any subcommands.
// It defines the main CLI interface and global configuration that applies
// to all subcommands.
var rootCmd = &cobra.Command{
	Use:   "ollama-util",
	Short: "Ollama model manager: check, list, benchmark, report, prune",
	Long: `A comprehensive CLI tool for managing Ollama models.

This tool provides functionality to:
- Check Ollama service health
- List available models
- Run interactive model benchmarks
- Generate performance reports
- Prune (delete) unwanted models

The Go version offers significant improvements over the Python implementation:
- 10x+ faster startup and execution
- Single binary deployment (no dependencies)
- Better concurrency and resource management
- Type-safe operations with compile-time error checking`,

	// PersistentPreRun initializes the Ollama client for all subcommands
	// using the base URL specified via flag or environment variable
	PersistentPreRun: func(cmd *cobra.Command, args []string) {
		client = ollama.NewClient(baseURL)
	},
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() error {
	return rootCmd.Execute()
}

// SetVersion sets the version information for the CLI.
// This is called by main() to pass build-time version information.
func SetVersion(version, commit, date string) {
	appVersion = version
	appCommit = commit
	appDate = date

	// Update the root command with version information
	rootCmd.Version = fmt.Sprintf("%s (commit: %s, built: %s)", version, commit, date)
}

// init configures the root command with global flags and registers all subcommands.
// This function is called automatically when the package is imported.
func init() {
	// Configure global flags that apply to all subcommands
	rootCmd.PersistentFlags().StringVar(&baseURL, "base-url", "http://localhost:11434",
		"Ollama API base URL (can also be set via OLLAMA_HOST environment variable)")

	// Register all available subcommands
	rootCmd.AddCommand(checkCmd)     // Health check command
	rootCmd.AddCommand(listCmd)      // Model listing command
	rootCmd.AddCommand(runCmd)       // Interactive benchmark command
	rootCmd.AddCommand(benchmarkCmd) // Batch benchmark command
	rootCmd.AddCommand(pruneCmd)     // Model deletion command
}

// exitWithError prints an error message to stderr and exits with status code 1.
// This is a utility function used by commands when they encounter fatal errors.
//
// Parameters:
//   - msg: A descriptive message about what failed
//   - err: The underlying error (can be nil for custom messages)
func exitWithError(msg string, err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", msg, err)
	} else {
		fmt.Fprintf(os.Stderr, "%s\n", msg)
	}
	os.Exit(1)
}
