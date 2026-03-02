package cmd

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/cobra"
)

// listCmd represents the list command
var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List all models",
	Long:  `List all available models from Ollama.`,
	Run: func(cmd *cobra.Command, args []string) {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		models, err := client.ListModels(ctx)
		if err != nil {
			exitWithError("Failed to list models", err)
		}

		if len(models) == 0 {
			fmt.Println("No models found.")
			return
		}

		fmt.Printf("Found %d model(s):\n", len(models))
		for _, model := range models {
			size := ""
			if model.Size > 0 {
				size = fmt.Sprintf("  (%s)", formatSize(model.Size))
			}
			fmt.Printf("  %s%s\n", model.Name, size)
		}
	},
}

// formatSize formats bytes into human readable string
func formatSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}
