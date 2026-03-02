package cmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/yourusername/ollama-util/internal/benchmark"
	"github.com/yourusername/ollama-util/internal/system"
)

var (
	benchmarkPrompt   string
	benchmarkOutput   string
	nvidiaSMIInterval float64
)

// benchmarkCmd represents the benchmark command
var benchmarkCmd = &cobra.Command{
	Use:   "benchmark",
	Short: "Run all models, benchmark, and print report",
	Long: `Run benchmarks on all available models and generate a performance report.

This command will:
- List all available models
- Run each model with the specified prompt
- Collect system performance metrics during execution
- Generate a comprehensive report`,
	Run: func(cmd *cobra.Command, args []string) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()

		// Get list of models
		models, err := client.ListModels(ctx)
		if err != nil {
			exitWithError("Failed to list models", err)
		}

		if len(models) == 0 {
			fmt.Println("No models found to benchmark.")
			return
		}

		// Extract model names
		var modelNames []string
		for _, model := range models {
			if model.Name != "" {
				modelNames = append(modelNames, model.Name)
			}
		}

		fmt.Printf("Benchmarking %d model(s) with prompt: %q\n", len(modelNames), benchmarkPrompt)

		// Create system monitor
		monitor := system.NewMonitor(true) // Enable GPU monitoring

		// Create benchmark runner
		runner := benchmark.NewRunner(client, monitor)

		// Run benchmarks
		results, err := runner.RunMultiple(ctx, modelNames, benchmarkPrompt)
		if err != nil {
			exitWithError("Benchmark failed", err)
		}

		// Display results
		fmt.Println("\n" + strings.Repeat("=", 60))
		fmt.Println("BENCHMARK RESULTS")
		fmt.Println(strings.Repeat("=", 60))

		for _, result := range results {
			fmt.Println(benchmark.FormatResult(result))
			fmt.Println(strings.Repeat("-", 40))
		}

		// Generate report if output file specified
		if benchmarkOutput != "" {
			if err := generateReport(results, benchmarkOutput); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: Failed to write report: %v\n", err)
			} else {
				fmt.Printf("\nReport written to: %s\n", benchmarkOutput)
			}
		}

		// Print summary
		printSummary(results)
	},
}

func init() {
	benchmarkCmd.Flags().StringVar(&benchmarkPrompt, "prompt", "Say exactly: OK", "Test prompt for benchmark")
	benchmarkCmd.Flags().StringVarP(&benchmarkOutput, "output", "o", "", "Write report to file (markdown format)")
	benchmarkCmd.Flags().Float64Var(&nvidiaSMIInterval, "nvidia-smi-interval", 10.0, "Run nvidia-smi every N seconds (currently unused)")
}

// generateReport generates a markdown report
func generateReport(results []*benchmark.Result, outputPath string) error {
	var sb strings.Builder

	sb.WriteString("# Ollama Benchmark Report\n\n")
	sb.WriteString(fmt.Sprintf("Generated: %s\n\n", time.Now().Format("2006-01-02 15:04:05")))

	// Summary table
	sb.WriteString("## Summary\n\n")
	sb.WriteString("| Model | Status | Duration | Tokens/sec | Output Tokens |\n")
	sb.WriteString("|-------|--------|----------|------------|---------------|\n")

	for _, result := range results {
		status := "✅"
		if !result.Success {
			status = "❌"
		}

		tokensPerSec := fmt.Sprintf("%.2f", result.TokensPerSecond)
		if result.TokensPerSecond == 0 {
			tokensPerSec = "-"
		}

		sb.WriteString(fmt.Sprintf("| %s | %s | %v | %s | %d |\n",
			result.ModelName, status, result.TotalDuration.Round(time.Millisecond),
			tokensPerSec, result.EvalCount))
	}

	// Detailed results
	sb.WriteString("\n## Detailed Results\n\n")

	for _, result := range results {
		sb.WriteString(fmt.Sprintf("### %s\n\n", result.ModelName))

		if result.Success {
			sb.WriteString("**Status:** ✅ Success\n\n")
			sb.WriteString(fmt.Sprintf("**Duration:** %v\n\n", result.TotalDuration.Round(time.Millisecond)))

			if result.TokensPerSecond > 0 {
				sb.WriteString(fmt.Sprintf("**Tokens/sec:** %.2f\n\n", result.TokensPerSecond))
			}

			sb.WriteString(fmt.Sprintf("**Prompt:** %s\n\n", result.Prompt))
			sb.WriteString(fmt.Sprintf("**Response:**\n```\n%s\n```\n\n", result.Response))
		} else {
			sb.WriteString("**Status:** ❌ Failed\n\n")
			if result.Error != "" {
				sb.WriteString(fmt.Sprintf("**Error:** %s\n\n", result.Error))
			}
		}

		// System info
		if result.SystemInfo != nil {
			sb.WriteString("**System Info:**\n```\n")
			sb.WriteString(system.FormatSystemInfo(result.SystemInfo))
			sb.WriteString("```\n\n")
		}
	}

	// Ensure output directory exists
	dir := filepath.Dir(outputPath)
	if dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("creating output directory: %w", err)
		}
	}

	// Write report
	return os.WriteFile(outputPath, []byte(sb.String()), 0644)
}

// printSummary prints a summary of the benchmark results
func printSummary(results []*benchmark.Result) {
	successful := 0
	failed := 0
	totalDuration := time.Duration(0)
	totalTokens := 0
	totalTokenTime := time.Duration(0)

	for _, result := range results {
		if result.Success {
			successful++
			totalDuration += result.TotalDuration
			if result.EvalCount > 0 && result.EvalDuration > 0 {
				totalTokens += result.EvalCount
				totalTokenTime += result.EvalDuration
			}
		} else {
			failed++
		}
	}

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("SUMMARY")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Total models: %d\n", len(results))
	fmt.Printf("Successful: %d\n", successful)
	fmt.Printf("Failed: %d\n", failed)

	if successful > 0 {
		avgDuration := totalDuration / time.Duration(successful)
		fmt.Printf("Average duration: %v\n", avgDuration.Round(time.Millisecond))

		if totalTokens > 0 && totalTokenTime > 0 {
			avgTokensPerSec := float64(totalTokens) / totalTokenTime.Seconds()
			fmt.Printf("Average tokens/sec: %.2f\n", avgTokensPerSec)
		}
	}
}
