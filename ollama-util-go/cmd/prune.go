package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

var (
	pruneModels string
)

// pruneCmd represents the prune command
var pruneCmd = &cobra.Command{
	Use:   "prune [MODEL_NAME...]",
	Short: "Delete (prune) models by name",
	Long: `Delete one or more models from Ollama.

You can specify model names as positional arguments or use the --models flag
with a comma-separated list.

Examples:
  ollama-util prune llama2
  ollama-util prune llama2 codellama
  ollama-util prune --models llama2,codellama,mistral`,
	Run: func(cmd *cobra.Command, args []string) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		// Collect model names from args and --models flag
		var modelNames []string

		// Add positional arguments
		for _, arg := range args {
			if strings.TrimSpace(arg) != "" {
				modelNames = append(modelNames, strings.TrimSpace(arg))
			}
		}

		// Add comma-separated models from --models flag
		if pruneModels != "" {
			for _, model := range strings.Split(pruneModels, ",") {
				if trimmed := strings.TrimSpace(model); trimmed != "" {
					modelNames = append(modelNames, trimmed)
				}
			}
		}

		if len(modelNames) == 0 {
			fmt.Fprintf(os.Stderr, "No model names provided. Use: prune MODEL [MODEL ...] or --models a,b,c\n")
			os.Exit(1)
		}

		fmt.Printf("Deleting %d model(s)...\n", len(modelNames))

		var failed []string
		var succeeded []string

		for _, modelName := range modelNames {
			fmt.Printf("Deleting %s... ", modelName)

			if err := client.DeleteModel(ctx, modelName); err != nil {
				fmt.Printf("❌ Failed: %v\n", err)
				failed = append(failed, modelName)
			} else {
				fmt.Println("✅ Success")
				succeeded = append(succeeded, modelName)
			}
		}

		// Print summary
		fmt.Println(strings.Repeat("-", 40))
		if len(succeeded) > 0 {
			fmt.Printf("✅ Successfully deleted: %s\n", strings.Join(succeeded, ", "))
		}
		if len(failed) > 0 {
			fmt.Printf("❌ Failed to delete: %s\n", strings.Join(failed, ", "))
			os.Exit(1)
		}

		fmt.Printf("Done! Deleted %d model(s)\n", len(succeeded))
	},
}

func init() {
	pruneCmd.Flags().StringVarP(&pruneModels, "models", "m", "", "Comma-separated model names to delete")
}
