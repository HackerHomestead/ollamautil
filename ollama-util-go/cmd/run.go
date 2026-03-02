package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"github.com/yourusername/ollama-util/internal/benchmark"
	"github.com/yourusername/ollama-util/internal/system"
)

var (
	runPrompt string
	runModels string
	runOutput string
	runQuiet  bool
	runPause  float64
)

// runCmd represents the run command
var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Interactive: select models and run benchmark with verbose output",
	Long: `Run interactive benchmarks with real-time output.

This command allows you to:
- Select specific models to benchmark (or run all if none specified)
- See real-time streaming output from each model
- Get detailed performance metrics
- Save results to a report file

Examples:
  ollama-util run
  ollama-util run --models llama2,codellama
  ollama-util run --prompt "write a python function" --output report.md`,
	Run: func(cmd *cobra.Command, args []string) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()

		// Get list of models
		models, err := client.ListModels(ctx)
		if err != nil {
			exitWithError("Failed to list models", err)
		}

		if len(models) == 0 {
			fmt.Println("No models found.")
			return
		}

		// Determine which models to run
		var modelNames []string
		if runModels != "" {
			// Use specified models
			for _, model := range strings.Split(runModels, ",") {
				if trimmed := strings.TrimSpace(model); trimmed != "" {
					modelNames = append(modelNames, trimmed)
				}
			}
		} else {
			// Use all models
			for _, model := range models {
				if model.Name != "" {
					modelNames = append(modelNames, model.Name)
				}
			}
		}

		if len(modelNames) == 0 {
			fmt.Println("No valid models to run.")
			return
		}

		// Use default prompt if none provided
		if runPrompt == "" {
			runPrompt = "write hello world in lisp"
		}

		fmt.Printf("Running %d model(s) with prompt: %q\n", len(modelNames), runPrompt)
		fmt.Println(strings.Repeat("=", 60))

		// Create system monitor
		monitor := system.NewMonitor(true) // Enable GPU monitoring

		// Create benchmark runner
		runner := benchmark.NewRunner(client, monitor)

		var results []*benchmark.Result

		// Run each model
		for i, modelName := range modelNames {
			fmt.Printf("\n[%d/%d] Running %s...\n", i+1, len(modelNames), modelName)

			if !runQuiet {
				fmt.Println(strings.Repeat("-", 40))
				fmt.Printf("Model: %s\n", modelName)
				fmt.Printf("Prompt: %s\n", runPrompt)
				fmt.Println("Response:")
			}

			// Run with streaming to show real-time output
			var responseText strings.Builder
			result, err := runner.RunSingleStreaming(ctx, modelName, runPrompt, func(chunk string) {
				if !runQuiet {
					fmt.Print(chunk)
				}
				responseText.WriteString(chunk)
			})

			if err != nil {
				fmt.Printf("\n❌ Error: %v\n", err)
				continue
			}

			results = append(results, result)

			if !runQuiet {
				fmt.Printf("\n\n✅ Completed in %v", result.TotalDuration.Round(time.Millisecond))
				if result.TokensPerSecond > 0 {
					fmt.Printf(" (%.2f tokens/sec)", result.TokensPerSecond)
				}
				fmt.Println()
			}

			// Pause between models if specified
			if runPause > 0 && i < len(modelNames)-1 {
				fmt.Printf("\nPausing for %.1f seconds before next model...\n", runPause)
				time.Sleep(time.Duration(runPause * float64(time.Second)))
			}
		}

		// Print summary
		fmt.Println(strings.Repeat("=", 60))
		fmt.Println("SUMMARY")
		fmt.Println(strings.Repeat("=", 60))

		successful := 0
		failed := 0
		totalDuration := time.Duration(0)

		for _, result := range results {
			if result.Success {
				successful++
				totalDuration += result.TotalDuration
				fmt.Printf("✅ %s - %v", result.ModelName, result.TotalDuration.Round(time.Millisecond))
				if result.TokensPerSecond > 0 {
					fmt.Printf(" (%.2f tokens/sec)", result.TokensPerSecond)
				}
				fmt.Println()
			} else {
				failed++
				fmt.Printf("❌ %s - %s\n", result.ModelName, result.Error)
			}
		}

		fmt.Printf("\nTotal: %d models, %d successful, %d failed\n", len(results), successful, failed)
		if successful > 0 {
			avgDuration := totalDuration / time.Duration(successful)
			fmt.Printf("Average duration: %v\n", avgDuration.Round(time.Millisecond))
		}

		// Generate report if output file specified
		if runOutput != "" {
			if err := generateReport(results, runOutput); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: Failed to write report: %v\n", err)
			} else {
				fmt.Printf("\nReport written to: %s\n", runOutput)
			}
		}
	},
}

func init() {
	runCmd.Flags().StringVar(&runPrompt, "prompt", "", "Test prompt for each model (if omitted, uses default)")
	runCmd.Flags().StringVarP(&runModels, "models", "m", "", "Comma-separated model names; if not set, run all models")
	runCmd.Flags().StringVarP(&runOutput, "output", "o", "", "Write results to a markdown file")
	runCmd.Flags().BoolVarP(&runQuiet, "quiet", "q", false, "Suppress live progress UI; print minimal output only")
	runCmd.Flags().Float64VarP(&runPause, "pause", "p", 15.0, "Pause between models (seconds); use 0 to disable")
}
