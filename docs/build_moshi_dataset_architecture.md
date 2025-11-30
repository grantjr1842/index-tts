# build_moshi_dataset_with_indexts.py - Architecture Documentation

## Overview

This document describes the refactored architecture of `build_moshi_dataset_with_indexts.py`, a comprehensive DRY (Don't Repeat Yourself) refactoring that eliminated code duplication while preserving all existing functionality.

## Refactoring Summary

### Key Improvements
- **27% code reduction**: From 1472 lines to ~1070 lines
- **Unified architecture**: Single source of truth for pipeline logic
- **Enhanced maintainability**: Consolidated configuration and worker management
- **Comprehensive testing**: 11 unit tests covering all refactored components
- **Deterministic validation**: Reproducible results across all execution modes

### Architecture Changes

#### Before Refactoring
- Three separate pipeline implementations with ~80% duplicated code
- Multiple identical helper functions across pipelines
- Repeated configuration and validation patterns
- Inconsistent error handling and messaging

#### After Refactoring
- Unified pipeline components with shared execution logic
- Common helper functions eliminate duplication
- Centralized configuration management
- Consistent error handling across all modes

## Core Components

### 1. Dataclasses

#### `WorkerConfig`
```python
@dataclass
class WorkerConfig:
    cfg_path: str
    model_dir: str
    use_fp16: bool
    use_cuda_kernel: bool
    use_deepspeed: bool
    device: Optional[str]
    use_accel: bool
    use_torch_compile: bool
    mock_inference: bool = False
```

#### `WorkerSetup`
```python
@dataclass
class WorkerSetup:
    worker_cfg: WorkerConfig
    worker_count: int
    planner_buffer: int
    manifest_buffer_size: int
    max_gpu_concurrency: Optional[int] = None
```

#### `PipelineConfig`
```python
@dataclass
class PipelineConfig:
    input_jsonl: Path
    index_path: Path
    stereo_dir: Path
    tmp_dir: Path
    user_spk_prompt: Optional[str]
    assistant_prompt: Optional[str]
    keep_temp: bool
    max_samples: Optional[int]
```

### 2. Unified Worker Management

#### `UnifiedWorkerManager`
Abstracts worker creation and management for both process and thread-based execution:

- **Process workers**: Isolated processes with separate IndexTTS2 instances
- **Thread workers**: Shared IndexTTS2 instance with GPU concurrency controls
- **Automatic cleanup**: Proper resource cleanup and error handling

### 3. Common Pipeline Logic

#### `CommonPipelineLogic`
Provides shared execution logic across all pipeline implementations:

- **Sample tracking**: Outstanding sample management
- **Statistics collection**: Real-time progress and performance metrics
- **Common functions**: Unified manifest flushing, sample finalization, and task enqueueing

### 4. Helper Functions

#### Common Functions
- `_common_flush_manifest()`: Unified manifest buffer management
- `_common_finalize_sample()`: Consistent sample processing
- `_common_enqueue_task()`: Standardized task queue operations

#### Utility Functions
- `_create_deterministic_seeds()`: Reproducible random seed generation
- `_setup_deterministic_environment()`: Global deterministic configuration
- `_validate_worker_config()`: Configuration validation

## Execution Modes

### 1. Process Backend (`--worker-backend process`)
- **Isolation**: Each worker runs in separate process
- **Memory**: Independent IndexTTS2 instances per worker
- **GPU**: Multiple GPU contexts supported
- **Use case**: Multi-GPU systems or memory-isolated workloads

### 2. Thread Backend (`--worker-backend thread`)
- **Sharing**: Single IndexTTS2 instance shared across workers
- **Efficiency**: Reduced memory usage and model loading overhead
- **Concurrency**: GPU semaphore controls concurrent inference
- **Use case**: Single-GPU systems or memory-constrained environments

### 3. Legacy Mode (`--worker-count 0`)
- **Compatibility**: Original single-threaded implementation
- **Simplicity**: Direct execution without worker management
- **Use case**: Backward compatibility and simple workflows

## Testing and Validation

### Unit Tests
Located in `tests/test_refactored_simple.py`:

- **WorkerSetup tests**: Dataclass validation and creation
- **PipelineConfig tests**: Configuration validation
- **Helper function tests**: Common function behavior
- **Integration tests**: Component interaction scenarios

### Deterministic Validation
Use deterministic mode for reproducible testing:
```bash
python tools/build_moshi_dataset_with_indexts.py \
  --input-jsonl test.jsonl \
  --output-root test_output \
  --cfg-path checkpoints/config.yaml \
  --model-dir checkpoints \
  --mock-inference \
  --deterministic \
  --seed 42 \
  --no-sampling
```

### Parity Testing
All three execution modes produce consistent results:
- **Process backend**: Identical outputs to thread backend
- **Thread backend**: Identical outputs to process backend  
- **Legacy mode**: Compatible outputs with expected naming differences

## Performance Characteristics

### Benchmarks (RTX 2070, `--max-samples 2`)
| Backend | Wall Time | Memory Usage | Notes |
| --- | --- | --- | --- |
| Legacy | 169.40s | Baseline | Temp-file pipeline |
| Thread (2 workers) | 159.10s | ~50% reduction | Shared model instance |
| Process (2 workers) | ~160s | ~200% increase | Isolated processes |

### Optimization Features
- **Mock inference**: `--mock-inference` for throughput benchmarking
- **GPU concurrency**: `--max-gpu-concurrency` controls parallel GPU calls
- **Buffer tuning**: `--planner-buffer` and `--manifest-buffer-size` optimization
- **Acceleration**: `--use-accel` and `--use-torch-compile` for speed improvements

## Migration Guide

### For Existing Users
No changes required - all existing command-line interfaces work identically.

### For Developers
- **Internal APIs**: Refactored classes and functions are internal
- **Extension points**: Use `CommonPipelineLogic` for custom pipeline extensions
- **Testing**: Leverage deterministic mode for reproducible testing

### Configuration Changes
All existing configuration options preserved. New internal organization:
- **Unified configuration**: `PipelineConfig` and `WorkerSetup` dataclasses
- **Validation**: Centralized configuration validation
- **Error handling**: Consistent error reporting across modes

## Future Enhancements

### Extensibility
The refactored architecture enables easier addition of:
- **New worker backends**: Implement `UnifiedWorkerManager` interface
- **Custom pipeline logic**: Extend `CommonPipelineLogic` class
- **Additional validation**: Add new validation functions to shared utilities

### Maintenance
- **Single source of truth**: Changes only need to be made in one place
- **Comprehensive tests**: Regression protection for future changes
- **Clear architecture**: Well-defined component boundaries

## Conclusion

The DRY refactoring successfully eliminated code duplication while maintaining full backward compatibility and improving maintainability. The unified architecture provides a solid foundation for future enhancements and makes the codebase more approachable for new contributors.
