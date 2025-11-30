#!/usr/bin/env python
"""
Simplified tests for refactored components in build_moshi_dataset_with_indexts.py

Tests the unified classes and functions without triggering multiprocessing.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the components we're testing
from tools.build_moshi_dataset_with_indexts import (
    WorkerSetup,
    PipelineConfig,
    WorkerConfig,
    SynthesisTask,
    SynthesisResult,
    ManifestEntry,
    _common_flush_manifest,
    _common_finalize_sample,
    _common_enqueue_task,
    _create_deterministic_seeds,
    _setup_deterministic_environment,
    _validate_worker_config,
    _write_stereo,
)


class TestWorkerSetup(unittest.TestCase):
    """Test the WorkerSetup dataclass"""
    
    def test_worker_setup_creation(self):
        """Test WorkerSetup can be created with valid parameters"""
        config = Mock(spec=WorkerConfig)
        setup = WorkerSetup(
            worker_cfg=config,
            worker_count=4,
            planner_buffer=10,
            manifest_buffer_size=100
        )
        self.assertEqual(setup.worker_cfg, config)
        self.assertEqual(setup.worker_count, 4)
        self.assertEqual(setup.planner_buffer, 10)
    
    def test_worker_setup_post_init_validation(self):
        """Test WorkerSetup validates planner_buffer in __post_init__"""
        config = Mock(spec=WorkerConfig)
        
        # Test planner_buffer gets corrected to minimum 1
        setup = WorkerSetup(
            worker_cfg=config,
            worker_count=4,
            planner_buffer=0,  # Will be set to 1 in __post_init__
            manifest_buffer_size=100
        )
        self.assertEqual(setup.planner_buffer, 1)  # Should be corrected


class TestPipelineConfig(unittest.TestCase):
    """Test the PipelineConfig dataclass"""
    
    def test_pipeline_config_creation(self):
        """Test PipelineConfig can be created with valid parameters"""
        config = PipelineConfig(
            input_jsonl=Path("test.jsonl"),
            index_path=Path("index.txt"),
            stereo_dir=Path("stereo"),
            tmp_dir=Path("tmp"),
            user_spk_prompt=None,
            assistant_prompt=None,
            keep_temp=False,
            max_samples=None
        )
        self.assertEqual(config.input_jsonl, Path("test.jsonl"))
        self.assertEqual(config.stereo_dir, Path("stereo"))
    
    def test_pipeline_config_validation(self):
        """Test PipelineConfig validation"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Test valid config
            config = PipelineConfig(
                input_jsonl=tmp_path / "test.jsonl",
                index_path=tmp_path / "index.txt",
                stereo_dir=tmp_path / "stereo",
                tmp_dir=tmp_path / "tmp",
                user_spk_prompt=None,
                assistant_prompt=None,
                keep_temp=False,
                max_samples=None
            )
            # Create the file for validation
            config.input_jsonl.touch()
            config.validate()  # Should not raise
            
            # Test invalid max_samples
            config = PipelineConfig(
                input_jsonl=tmp_path / "test.jsonl",
                index_path=tmp_path / "index.txt",
                stereo_dir=tmp_path / "stereo",
                tmp_dir=tmp_path / "tmp",
                user_spk_prompt=None,
                assistant_prompt=None,
                keep_temp=False,
                max_samples=0
            )
            config.input_jsonl.touch()
            with self.assertRaises(ValueError):
                config.validate()


class TestCommonHelperFunctions(unittest.TestCase):
    """Test the common helper functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manifest_file = self.temp_dir / "test_manifest.txt"
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_common_flush_manifest(self):
        """Test _common_flush_manifest function"""
        # Create test manifest entries
        entries = [
            {"key": "value1"},
            {"key": "value2"}
        ]
        
        # Write to manifest file
        with open(self.manifest_file, 'w') as f:
            _common_flush_manifest(entries, f)
        
        # Check file was written
        self.assertTrue(self.manifest_file.exists())
        
        # Check content
        with open(self.manifest_file, 'r') as f:
            content = f.read()
            self.assertIn("value1", content)
            self.assertIn("value2", content)
    
    def test_common_enqueue_task(self):
        """Test _common_enqueue_task function"""
        mock_queue = Mock()
        mock_stop_event = Mock()
        mock_stop_event.is_set.return_value = False
        tasks_enqueued_ref = [0]
        
        task = SynthesisTask(
            sample_id="test",
            role="user",
            text="test text",
            spk_audio_prompt=None,
            emo_audio_prompt=None,
            emo_vector=None
        )
        
        result = _common_enqueue_task(task, mock_queue, mock_stop_event, tasks_enqueued_ref)
        
        self.assertTrue(result)
        mock_queue.put.assert_called_once_with(task, timeout=0.5)
        self.assertEqual(tasks_enqueued_ref[0], 1)
    
    def test_create_deterministic_seeds(self):
        """Test _create_deterministic_seeds function"""
        seeds = _create_deterministic_seeds(42)
        
        # Should return dict with expected keys
        self.assertIn('numpy_seed', seeds)
        self.assertIn('torch_seed', seeds)
        self.assertIn('random_seed', seeds)
        
        # Should be deterministic
        seeds2 = _create_deterministic_seeds(42)
        self.assertEqual(seeds, seeds2)
    
    def test_validate_worker_config(self):
        """Test _validate_worker_config function"""
        # Test valid config
        config = Mock(spec=WorkerConfig)
        
        # Should not raise (the actual validation is minimal)
        _validate_worker_config(config)
    
    def test_write_stereo_function(self):
        """Test _write_stereo function"""
        # Create test audio data
        import numpy as np
        left_audio = np.random.rand(1000).astype(np.float32)
        right_audio = np.random.rand(1000).astype(np.float32)
        output_path = self.temp_dir / "test_stereo.wav"
        
        # Write stereo file
        duration = _write_stereo(left_audio, right_audio, 22050, output_path)
        
        # Check file was created
        self.assertTrue(output_path.exists())
        self.assertGreater(duration, 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for refactored components working together"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_worker_setup_with_pipeline_config(self):
        """Test WorkerSetup works with PipelineConfig"""
        config = PipelineConfig(
            input_jsonl=self.temp_dir / "test.jsonl",
            index_path=self.temp_dir / "index.txt",
            stereo_dir=self.temp_dir / "stereo",
            tmp_dir=self.temp_dir / "tmp",
            user_spk_prompt=None,
            assistant_prompt=None,
            keep_temp=False,
            max_samples=None
        )
        
        worker_config = Mock(spec=WorkerConfig)
        setup = WorkerSetup(
            worker_cfg=worker_config,
            worker_count=4,
            planner_buffer=10,
            manifest_buffer_size=100
        )
        
        self.assertEqual(setup.worker_count, 4)
    
    def test_deterministic_seeds_integration(self):
        """Test deterministic seeds work with environment setup"""
        seeds = _create_deterministic_seeds(123)
        
        # Should not raise when setting up environment
        _setup_deterministic_environment(seeds)
        
        # Should be deterministic
        seeds2 = _create_deterministic_seeds(123)
        self.assertEqual(seeds, seeds2)
        
        # Different base seeds should produce different results
        seeds3 = _create_deterministic_seeds(456)
        self.assertNotEqual(seeds, seeds3)


if __name__ == "__main__":
    unittest.main()
