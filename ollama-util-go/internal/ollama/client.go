package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// HTTPClient implements the Client interface using HTTP requests to communicate
// with the Ollama REST API. It handles all HTTP-level concerns including
// request/response marshaling, error handling, and connection management.
type HTTPClient struct {
	baseURL string       // Ollama API base URL (e.g., "http://localhost:11434")
	client  *http.Client // Underlying HTTP client with configured timeouts
}

// NewClient creates a new Ollama HTTP client with sensible defaults.
// If baseURL is empty, it defaults to "http://localhost:11434" which is
// the standard Ollama service endpoint.
//
// The client is configured with a 5-minute timeout for most operations,
// though individual operations may override this for longer-running tasks
// like model generation.
func NewClient(baseURL string) *HTTPClient {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	return &HTTPClient{
		baseURL: strings.TrimSuffix(baseURL, "/"),
		client: &http.Client{
			Timeout: 5 * time.Minute, // Default timeout for most operations
		},
	}
}

// Health checks if the Ollama service is running
func (c *HTTPClient) Health(ctx context.Context) error {
	_, err := c.Version(ctx)
	return err
}

// Version gets the version information from Ollama
func (c *HTTPClient) Version(ctx context.Context) (*VersionResponse, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/api/version", nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	var version VersionResponse
	if err := json.NewDecoder(resp.Body).Decode(&version); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return &version, nil
}

// ListModels lists all available models
func (c *HTTPClient) ListModels(ctx context.Context) ([]Model, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	var listResp ListModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}

	return listResp.Models, nil
}

// Generate generates text using the specified model (non-streaming)
func (c *HTTPClient) Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error) {
	// Force non-streaming
	req.Stream = false

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/generate", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Use longer timeout for generation
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	// Ollama may return NDJSON even for non-streaming requests
	// We need to handle both single JSON and NDJSON responses
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}

	// Try parsing as single JSON first
	var genResp GenerateResponse
	if err := json.Unmarshal(body, &genResp); err == nil {
		return &genResp, nil
	}

	// If that fails, try parsing as NDJSON
	return c.parseNDJSONResponse(body)
}

// GenerateStream generates text using the specified model (streaming)
func (c *HTTPClient) GenerateStream(ctx context.Context, req GenerateRequest) <-chan StreamChunk {
	ch := make(chan StreamChunk, 1)

	go func() {
		defer close(ch)

		// Force streaming
		req.Stream = true

		jsonData, err := json.Marshal(req)
		if err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("marshaling request: %w", err)}
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/generate", bytes.NewBuffer(jsonData))
		if err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("creating request: %w", err)}
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")

		// Use longer timeout for streaming
		client := &http.Client{Timeout: 10 * time.Minute}
		resp, err := client.Do(httpReq)
		if err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("executing request: %w", err)}
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			ch <- StreamChunk{Error: fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))}
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.TrimSpace(line) == "" {
				continue
			}

			var genResp GenerateResponse
			if err := json.Unmarshal([]byte(line), &genResp); err != nil {
				ch <- StreamChunk{Error: fmt.Errorf("parsing stream chunk: %w", err)}
				return
			}

			select {
			case ch <- StreamChunk{Response: &genResp}:
			case <-ctx.Done():
				return
			}

			// If this is the final chunk, we're done
			if genResp.Done {
				return
			}
		}

		if err := scanner.Err(); err != nil {
			ch <- StreamChunk{Error: fmt.Errorf("reading stream: %w", err)}
		}
	}()

	return ch
}

// DeleteModel deletes a model by name
func (c *HTTPClient) DeleteModel(ctx context.Context, name string) error {
	req := DeleteRequest{Name: name}
	jsonData, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "DELETE", c.baseURL+"/api/delete", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("executing request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// parseNDJSONResponse parses NDJSON response and merges into single response
func (c *HTTPClient) parseNDJSONResponse(body []byte) (*GenerateResponse, error) {
	lines := strings.Split(string(body), "\n")
	var merged GenerateResponse
	var fullResponse strings.Builder

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var chunk GenerateResponse
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue // Skip invalid JSON lines
		}

		// Accumulate response text
		if chunk.Response != "" {
			fullResponse.WriteString(chunk.Response)
		}

		// Use the last chunk's metadata
		merged = chunk
	}

	// Set the complete response
	merged.Response = fullResponse.String()
	return &merged, nil
}
