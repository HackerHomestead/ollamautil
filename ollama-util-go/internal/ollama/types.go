// Package ollama provides types and client implementation for interacting with the Ollama HTTP API.
//
// This package contains all the data structures that map to Ollama's REST API responses,
// as well as the HTTP client implementation that handles communication with the Ollama service.
// The types are designed to be both JSON-serializable and suitable for use in Go programs
// with proper type safety and null handling.
package ollama

import (
	"context"
	"time"
)

// Model represents a model returned by the Ollama API (/api/tags endpoint).
// This structure contains all metadata about an installed model including
// size, modification time, and configuration details.
type Model struct {
	Name       string         `json:"name"`                  // Model name (e.g., "llama2:7b")
	Size       int64          `json:"size,omitempty"`        // Model size in bytes
	Digest     string         `json:"digest,omitempty"`      // Model content hash
	Details    *ModelDetails  `json:"details,omitempty"`     // Extended model information
	ExpiresAt  *time.Time     `json:"expires_at,omitempty"`  // When model expires from cache
	SizeVRAM   int64          `json:"size_vram,omitempty"`   // VRAM usage in bytes
	ModifiedAt *time.Time     `json:"modified_at,omitempty"` // Last modification time
	Parameters map[string]any `json:"parameters,omitempty"`  // Model parameters/config
	Template   string         `json:"template,omitempty"`    // Prompt template
	System     string         `json:"system,omitempty"`      // System message
}

// ModelDetails contains detailed information about a model's architecture and capabilities.
// This is populated by Ollama for models that have been fully loaded and analyzed.
type ModelDetails struct {
	ParentModel       string   `json:"parent_model,omitempty"`       // Base model this was derived from
	Format            string   `json:"format,omitempty"`             // Model format (e.g., "gguf")
	Family            string   `json:"family,omitempty"`             // Model family (e.g., "llama")
	Families          []string `json:"families,omitempty"`           // All related model families
	ParameterSize     string   `json:"parameter_size,omitempty"`     // Number of parameters (e.g., "7B")
	QuantizationLevel string   `json:"quantization_level,omitempty"` // Quantization level (e.g., "Q4_0")
}

// GenerateRequest represents a request to generate text via the /api/generate endpoint.
// This structure supports both streaming and non-streaming text generation with
// various configuration options for controlling model behavior.
type GenerateRequest struct {
	Model    string         `json:"model"`              // Model name to use for generation
	Prompt   string         `json:"prompt"`             // Input prompt for text generation
	System   string         `json:"system,omitempty"`   // System message to set context
	Template string         `json:"template,omitempty"` // Custom prompt template
	Context  []int          `json:"context,omitempty"`  // Context tokens from previous generation
	Stream   bool           `json:"stream,omitempty"`   // Whether to stream response chunks
	Raw      bool           `json:"raw,omitempty"`      // Use raw prompt without formatting
	Format   string         `json:"format,omitempty"`   // Response format (e.g., "json")
	Options  map[string]any `json:"options,omitempty"`  // Model-specific options (temperature, etc.)
}

// GenerateResponse represents a response from the generate API (/api/generate).
// Contains the generated text along with performance timing information that
// can be used for benchmarking and performance analysis.
type GenerateResponse struct {
	Model              string    `json:"model"`                          // Model used for generation
	CreatedAt          time.Time `json:"created_at"`                     // Response timestamp
	Response           string    `json:"response"`                       // Generated text content
	Done               bool      `json:"done"`                           // Whether generation is complete
	Context            []int     `json:"context,omitempty"`              // Updated context tokens
	TotalDuration      int64     `json:"total_duration,omitempty"`       // Total time in nanoseconds
	LoadDuration       int64     `json:"load_duration,omitempty"`        // Model loading time in nanoseconds
	PromptEvalCount    int       `json:"prompt_eval_count,omitempty"`    // Number of tokens in prompt
	PromptEvalDuration int64     `json:"prompt_eval_duration,omitempty"` // Prompt processing time in nanoseconds
	EvalCount          int       `json:"eval_count,omitempty"`           // Number of generated tokens
	EvalDuration       int64     `json:"eval_duration,omitempty"`        // Generation time in nanoseconds
}

// ListModelsResponse represents the response from the /api/tags endpoint.
// Contains an array of all models installed on the Ollama server.
type ListModelsResponse struct {
	Models []Model `json:"models"` // Array of installed models
}

// VersionResponse represents the response from the /api/version endpoint.
// Used for health checking and version compatibility verification.
type VersionResponse struct {
	Version string `json:"version"` // Ollama server version string
}

// DeleteRequest represents a request to delete a model via /api/delete.
// Only the model name is required to identify which model to remove.
type DeleteRequest struct {
	Name string `json:"name"` // Name of the model to delete
}

// StreamChunk represents a chunk of data from a streaming response.
// Used internally to handle streaming generation responses with proper
// error handling and type safety.
type StreamChunk struct {
	Response *GenerateResponse // Parsed response chunk (nil if error occurred)
	Error    error             // Error that occurred during streaming (nil if successful)
}

// Client interface defines the methods for interacting with Ollama HTTP API.
// This interface abstracts the HTTP client implementation to allow for easier
// testing and potential alternative implementations (e.g., gRPC, WebSocket).
//
// All methods accept a context.Context for proper cancellation and timeout handling.
// Methods return appropriate error types for robust error handling in calling code.
type Client interface {
	// Health checks if the Ollama service is running and accessible
	Health(ctx context.Context) error

	// Version retrieves the Ollama server version information
	Version(ctx context.Context) (*VersionResponse, error)

	// ListModels retrieves all models installed on the Ollama server
	ListModels(ctx context.Context) ([]Model, error)

	// Generate performs text generation with the specified model (non-streaming)
	Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error)

	// GenerateStream performs streaming text generation, returning a channel of response chunks
	GenerateStream(ctx context.Context, req GenerateRequest) <-chan StreamChunk

	// DeleteModel removes a model from the Ollama server
	DeleteModel(ctx context.Context, name string) error
}
