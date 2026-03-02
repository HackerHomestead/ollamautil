package cmd

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/cobra"
)

// checkCmd represents the check command
var checkCmd = &cobra.Command{
	Use:   "check",
	Short: "Check if Ollama is running",
	Long:  `Check if the Ollama service is running and accessible.`,
	Run: func(cmd *cobra.Command, args []string) {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		fmt.Printf("Checking Ollama at %s...\n", baseURL)

		if err := client.Health(ctx); err != nil {
			exitWithError("Ollama check failed", err)
		}

		fmt.Println("✅ Ollama is running.")

		// Try to get version info
		if version, err := client.Version(ctx); err == nil && version.Version != "" {
			fmt.Printf("Version: %s\n", version.Version)
		}
	},
}
