# Implementation Plan - Moshi Dataset Builder Optimization

[Overview]
Validate and document the optimized `tools/build_moshi_dataset_with_indexts.py` pipeline, which implements parallel synthesis and memory-based audio handling.

The implementation of the optimized builder (Issue #3) appears to be code-complete. This plan focuses on validating the implementation via regression testing against the legacy path, and updating project documentation to reflect the new capabilities.

[Types]
No changes to type definitions required.
Existing `InferenceResult` protocol and `SynthesisTask`/`SynthesisResult` dataclasses in `tools/build_moshi_dataset_with_indexts.py` are sufficient.

[Files]
Validate and Document existing files.

Detailed breakdown:
- `tools/build_moshi_dataset_with_indexts.py`: Verify functionality (already implemented).
- `README.md`: Update with usage instructions for the new builder script.
- `logs/regressions/`: Create this directory if it doesn't exist, to store regression test logs.

[Functions]
No function modifications required (validation only).

[Classes]
No class modifications required (validation only).

[Dependencies]
No new dependencies.

[Implementation Order]
Validate the existing implementation and document it.

1. Run regression test: Compare `tools/build_moshi_dataset_with_indexts.py --legacy-io` vs parallel execution on `examples/moshi_sample.jsonl`.
2. Update `README.md` with new CLI usage examples.
3. Commit/finalize.
