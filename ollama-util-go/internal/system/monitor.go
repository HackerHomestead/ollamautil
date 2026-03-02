// Package system provides system monitoring capabilities for collecting
// CPU, memory, and GPU information during benchmarking operations.
//
// This package uses gopsutil for cross-platform system information gathering
// and nvidia-smi for GPU monitoring when available. It's designed to provide
// comprehensive system metrics that can be included in benchmark reports.
package system

import (
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/shirou/gopsutil/v3/mem"
)

// SystemInfo represents comprehensive system information collected at a point in time.
// This structure is designed to capture all relevant system metrics that might
// affect model performance, including hardware specs and current resource usage.
type SystemInfo struct {
	OS           string     `json:"os"`             // Operating system (linux, darwin, windows)
	Architecture string     `json:"architecture"`   // CPU architecture (amd64, arm64, etc.)
	CPUCores     int        `json:"cpu_cores"`      // Number of logical CPU cores
	Memory       MemoryInfo `json:"memory"`         // Memory usage information
	GPUs         []GPUInfo  `json:"gpus,omitempty"` // GPU information (if available)
	Timestamp    time.Time  `json:"timestamp"`      // When this information was collected
}

// MemoryInfo represents current memory usage statistics.
// Values are provided in human-readable units (GB) for easier interpretation
// in reports and logging output.
type MemoryInfo struct {
	TotalGB     float64 `json:"total_gb"`     // Total system memory in GB
	AvailableGB float64 `json:"available_gb"` // Available memory in GB
	PercentUsed float64 `json:"percent_used"` // Memory utilization percentage
}

// GPUInfo represents detailed information about a GPU device.
// This information is collected via nvidia-smi and provides comprehensive
// metrics about GPU state, memory usage, and performance characteristics.
type GPUInfo struct {
	Name           string  `json:"name"`                         // GPU model name
	MemoryTotalMB  float64 `json:"memory_total_mb,omitempty"`    // Total VRAM in MB
	MemoryUsedMB   float64 `json:"memory_used_mb,omitempty"`     // Used VRAM in MB
	MemoryFreeMB   float64 `json:"memory_free_mb,omitempty"`     // Free VRAM in MB
	UtilizationGPU int     `json:"utilization_gpu,omitempty"`    // GPU utilization percentage
	UtilizationMem int     `json:"utilization_memory,omitempty"` // Memory utilization percentage
	Temperature    int     `json:"temperature,omitempty"`        // GPU temperature in Celsius
	DriverVersion  string  `json:"driver_version,omitempty"`     // NVIDIA driver version
}

// Monitor provides system monitoring capabilities
type Monitor struct {
	collectGPU bool
}

// NewMonitor creates a new system monitor
func NewMonitor(collectGPU bool) *Monitor {
	return &Monitor{
		collectGPU: collectGPU && isNvidiaSMIAvailable(),
	}
}

// Collect gathers current system information
func (m *Monitor) Collect(ctx context.Context) (*SystemInfo, error) {
	info := &SystemInfo{
		OS:           runtime.GOOS,
		Architecture: runtime.GOARCH,
		CPUCores:     runtime.NumCPU(),
		Timestamp:    time.Now(),
	}

	// Get memory info
	memInfo, err := mem.VirtualMemory()
	if err != nil {
		return nil, fmt.Errorf("getting memory info: %w", err)
	}

	info.Memory = MemoryInfo{
		TotalGB:     float64(memInfo.Total) / (1024 * 1024 * 1024),
		AvailableGB: float64(memInfo.Available) / (1024 * 1024 * 1024),
		PercentUsed: memInfo.UsedPercent,
	}

	// Get GPU info if requested
	if m.collectGPU {
		if gpus, err := m.collectGPUInfo(ctx); err == nil {
			info.GPUs = gpus
		}
	}

	return info, nil
}

// CollectContinuous collects system information at regular intervals
func (m *Monitor) CollectContinuous(ctx context.Context, interval time.Duration) <-chan *SystemInfo {
	ch := make(chan *SystemInfo, 1)

	go func() {
		defer close(ch)

		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		// Send initial reading
		if info, err := m.Collect(ctx); err == nil {
			select {
			case ch <- info:
			case <-ctx.Done():
				return
			}
		}

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if info, err := m.Collect(ctx); err == nil {
					select {
					case ch <- info:
					case <-ctx.Done():
						return
					}
				}
			}
		}
	}()

	return ch
}

// isNvidiaSMIAvailable checks if nvidia-smi is available
func isNvidiaSMIAvailable() bool {
	_, err := exec.LookPath("nvidia-smi")
	return err == nil
}

// collectGPUInfo collects GPU information using nvidia-smi
func (m *Monitor) collectGPUInfo(ctx context.Context) ([]GPUInfo, error) {
	if !isNvidiaSMIAvailable() {
		return nil, fmt.Errorf("nvidia-smi not available")
	}

	cmd := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,driver_version",
		"--format=csv,noheader,nounits")

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("running nvidia-smi: %w", err)
	}

	var gpus []GPUInfo
	lines := strings.Split(string(output), "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		parts := strings.Split(line, ", ")
		if len(parts) < 8 {
			continue
		}

		gpu := GPUInfo{
			Name:          strings.TrimSpace(parts[0]),
			DriverVersion: strings.TrimSpace(parts[7]),
		}

		// Parse numeric values
		if val, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64); err == nil {
			gpu.MemoryTotalMB = val
		}
		if val, err := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64); err == nil {
			gpu.MemoryUsedMB = val
		}
		if val, err := strconv.ParseFloat(strings.TrimSpace(parts[3]), 64); err == nil {
			gpu.MemoryFreeMB = val
		}
		if val, err := strconv.Atoi(strings.TrimSpace(parts[4])); err == nil {
			gpu.UtilizationGPU = val
		}
		if val, err := strconv.Atoi(strings.TrimSpace(parts[5])); err == nil {
			gpu.UtilizationMem = val
		}
		if val, err := strconv.Atoi(strings.TrimSpace(parts[6])); err == nil {
			gpu.Temperature = val
		}

		gpus = append(gpus, gpu)
	}

	return gpus, nil
}

// FormatSystemInfo formats system info for display
func FormatSystemInfo(info *SystemInfo) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("OS: %s/%s\n", info.OS, info.Architecture))
	sb.WriteString(fmt.Sprintf("CPU Cores: %d\n", info.CPUCores))
	sb.WriteString(fmt.Sprintf("Memory: %.1f GB total, %.1f GB available (%.1f%% used)\n",
		info.Memory.TotalGB, info.Memory.AvailableGB, info.Memory.PercentUsed))

	if len(info.GPUs) > 0 {
		sb.WriteString(fmt.Sprintf("GPUs: %d found\n", len(info.GPUs)))
		for i, gpu := range info.GPUs {
			sb.WriteString(fmt.Sprintf("  GPU %d: %s\n", i, gpu.Name))
			if gpu.MemoryTotalMB > 0 {
				sb.WriteString(fmt.Sprintf("    Memory: %.0f MB total, %.0f MB used (%.1f%%)\n",
					gpu.MemoryTotalMB, gpu.MemoryUsedMB, (gpu.MemoryUsedMB/gpu.MemoryTotalMB)*100))
			}
			if gpu.UtilizationGPU > 0 {
				sb.WriteString(fmt.Sprintf("    Utilization: %d%% GPU, %d%% Memory\n",
					gpu.UtilizationGPU, gpu.UtilizationMem))
			}
			if gpu.Temperature > 0 {
				sb.WriteString(fmt.Sprintf("    Temperature: %d°C\n", gpu.Temperature))
			}
		}
	}

	return sb.String()
}
