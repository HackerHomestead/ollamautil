package benchmark

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/yourusername/ollama-util/internal/ollama"
	"github.com/yourusername/ollama-util/internal/system"
)

// Result represents the result of benchmarking a single model
type Result struct {
	ModelName       string             `json:"model_name"`
	Prompt          string             `json:"prompt"`
	Response        string             `json:"response"`
	Success         bool               `json:"success"`
	Error           string             `json:"error,omitempty"`
	StartTime       time.Time          `json:"start_time"`
	EndTime         time.Time          `json:"end_time"`
	TotalDuration   time.Duration      `json:"total_duration"`
	LoadDuration    time.Duration      `json:"load_duration,omitempty"`
	EvalDuration    time.Duration      `json:"eval_duration,omitempty"`
	PromptEvalCount int                `json:"prompt_eval_count,omitempty"`
	EvalCount       int                `json:"eval_count,omitempty"`
	TokensPerSecond float64            `json:"tokens_per_second,omitempty"`
	SystemInfo      *system.SystemInfo `json:"system_info,omitempty"`
}

// Runner handles benchmarking operations
type Runner struct {
	client  ollama.Client
	monitor *system.Monitor
}

// NewRunner creates a new benchmark runner
func NewRunner(client ollama.Client, monitor *system.Monitor) *Runner {
	return &Runner{
		client:  client,
		monitor: monitor,
	}
}

// RunSingle benchmarks a single model with the given prompt
func (r *Runner) RunSingle(ctx context.Context, modelName, prompt string) (*Result, error) {
	result := &Result{
		ModelName: modelName,
		Prompt:    prompt,
		StartTime: time.Now(),
	}

	// Collect system info before benchmark
	if r.monitor != nil {
		if sysInfo, err := r.monitor.Collect(ctx); err == nil {
			result.SystemInfo = sysInfo
		}
	}

	// Create generate request
	req := ollama.GenerateRequest{
		Model:  modelName,
		Prompt: prompt,
		Stream: false,
	}

	// Run the generation
	resp, err := r.client.Generate(ctx, req)
	result.EndTime = time.Now()
	result.TotalDuration = result.EndTime.Sub(result.StartTime)

	if err != nil {
		result.Success = false
		result.Error = err.Error()
		return result, nil // Return result with error, don't fail
	}

	// Process successful response
	result.Success = true
	result.Response = strings.TrimSpace(resp.Response)

	// Extract timing information
	if resp.LoadDuration > 0 {
		result.LoadDuration = time.Duration(resp.LoadDuration) * time.Nanosecond
	}
	if resp.EvalDuration > 0 {
		result.EvalDuration = time.Duration(resp.EvalDuration) * time.Nanosecond
	}
	if resp.PromptEvalCount > 0 {
		result.PromptEvalCount = resp.PromptEvalCount
	}
	if resp.EvalCount > 0 {
		result.EvalCount = resp.EvalCount
	}

	// Calculate tokens per second
	if result.EvalCount > 0 && result.EvalDuration > 0 {
		result.TokensPerSecond = float64(result.EvalCount) / result.EvalDuration.Seconds()
	}

	return result, nil
}

// RunSingleStreaming benchmarks a single model with streaming
func (r *Runner) RunSingleStreaming(ctx context.Context, modelName, prompt string, onChunk func(chunk string)) (*Result, error) {
	result := &Result{
		ModelName: modelName,
		Prompt:    prompt,
		StartTime: time.Now(),
	}

	// Collect system info before benchmark
	if r.monitor != nil {
		if sysInfo, err := r.monitor.Collect(ctx); err == nil {
			result.SystemInfo = sysInfo
		}
	}

	// Create generate request
	req := ollama.GenerateRequest{
		Model:  modelName,
		Prompt: prompt,
		Stream: true,
	}

	// Run the streaming generation
	stream := r.client.GenerateStream(ctx, req)
	var responseBuilder strings.Builder
	var lastResp *ollama.GenerateResponse

	for chunk := range stream {
		if chunk.Error != nil {
			result.EndTime = time.Now()
			result.TotalDuration = result.EndTime.Sub(result.StartTime)
			result.Success = false
			result.Error = chunk.Error.Error()
			return result, nil
		}

		if chunk.Response != nil {
			lastResp = chunk.Response
			if chunk.Response.Response != "" {
				responseBuilder.WriteString(chunk.Response.Response)
				if onChunk != nil {
					onChunk(chunk.Response.Response)
				}
			}

			if chunk.Response.Done {
				break
			}
		}
	}

	result.EndTime = time.Now()
	result.TotalDuration = result.EndTime.Sub(result.StartTime)
	result.Success = true
	result.Response = strings.TrimSpace(responseBuilder.String())

	// Extract timing information from the final response
	if lastResp != nil {
		if lastResp.LoadDuration > 0 {
			result.LoadDuration = time.Duration(lastResp.LoadDuration) * time.Nanosecond
		}
		if lastResp.EvalDuration > 0 {
			result.EvalDuration = time.Duration(lastResp.EvalDuration) * time.Nanosecond
		}
		if lastResp.PromptEvalCount > 0 {
			result.PromptEvalCount = lastResp.PromptEvalCount
		}
		if lastResp.EvalCount > 0 {
			result.EvalCount = lastResp.EvalCount
		}

		// Calculate tokens per second
		if result.EvalCount > 0 && result.EvalDuration > 0 {
			result.TokensPerSecond = float64(result.EvalCount) / result.EvalDuration.Seconds()
		}
	}

	return result, nil
}

// RunMultiple benchmarks multiple models sequentially
func (r *Runner) RunMultiple(ctx context.Context, modelNames []string, prompt string) ([]*Result, error) {
	results := make([]*Result, 0, len(modelNames))

	for _, modelName := range modelNames {
		result, err := r.RunSingle(ctx, modelName, prompt)
		if err != nil {
			return results, fmt.Errorf("benchmarking model %s: %w", modelName, err)
		}
		results = append(results, result)
	}

	return results, nil
}

// FormatResult formats a benchmark result for display
func FormatResult(result *Result) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Model: %s\n", result.ModelName))
	if result.Success {
		sb.WriteString("Status: ✅ Success\n")
		sb.WriteString(fmt.Sprintf("Duration: %v\n", result.TotalDuration.Round(time.Millisecond)))

		if result.TokensPerSecond > 0 {
			sb.WriteString(fmt.Sprintf("Tokens/sec: %.2f\n", result.TokensPerSecond))
		}

		if result.EvalCount > 0 {
			sb.WriteString(fmt.Sprintf("Output tokens: %d\n", result.EvalCount))
		}

		if result.PromptEvalCount > 0 {
			sb.WriteString(fmt.Sprintf("Prompt tokens: %d\n", result.PromptEvalCount))
		}

		if result.LoadDuration > 0 {
			sb.WriteString(fmt.Sprintf("Load time: %v\n", result.LoadDuration.Round(time.Millisecond)))
		}

		sb.WriteString(fmt.Sprintf("Response: %s\n", truncateString(result.Response, 100)))
	} else {
		sb.WriteString("Status: ❌ Failed\n")
		if result.Error != "" {
			sb.WriteString(fmt.Sprintf("Error: %s\n", result.Error))
		}
	}

	return sb.String()
}

// truncateString truncates a string to maxLen characters, adding "..." if needed
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
