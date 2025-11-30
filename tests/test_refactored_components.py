#!/usr/bin/env python
"""
Comprehensive tests for refactored components in build_moshi_dataset_with_indexts.py

Tests the unified classes and functions introduced during DRY refactoring:
- WorkerSetup dataclass
- UnifiedWorkerManager class  
- PipelineConfig dataclass
- CommonPipelineLogic class
- Common helper functions
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the components we're testing
from tools.build_moshi_dataset_with_indexts import (
    WorkerSetup,
    UnifiedWorkerManager,
    PipelineConfig,
    CommonPipelineLogic,
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
    IndexTTS2Inferable,
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


class TestUnifiedWorkerManager(unittest.TestCase):
    """Test the UnifiedWorkerManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=WorkerConfig)
        self.setup = WorkerSetup(
            worker_cfg=self.config,
            worker_count=2,
            planner_buffer=10,
            manifest_buffer_size=100
        )
    
    def test_process_worker_manager_creation(self):
        """Test UnifiedWorkerManager with process workers"""
        manager = UnifiedWorkerManager(self.setup, use_processes=True)
        self.assertTrue(manager.use_processes)
        self.assertEqual(manager.setup, self.setup)
    
    def test_thread_worker_manager_creation(self):
        """Test UnifiedWorkerManager with thread workers"""
        manager = UnifiedWorkerManager(self.setup, use_processes=False)
        self.assertFalse(manager.use_processes)
        self.assertEqual(manager.setup, self.setup)
    
    @patch('multiprocessing.Process')
    def test_setup_process_workers(self, mock_process):
        """Test process worker setup"""
        manager = UnifiedWorkerManager(self.setup, use_processes=True)
        
        # Mock the target function
        mock_target = Mock()
        
        manager._setup_process_workers()
        
        # Should create num_workers processes
        self.assertEqual(mock_process.call_count, 2)
    
    @patch('threading.Thread')
    def test_setup_thread_workers(self, mock_thread):
        """Test thread worker setup"""
        manager = UnifiedWorkerManager(self.setup, use_processes=False)
        
        manager._setup_thread_workers()
        
        # Should create num_workers threads
        self.assertEqual(mock_thread.call_count, 2)


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
        # Test valid config
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
        # Create the file for validation
        config.input_jsonl.touch()
        config.validate()  # Should not raise
        
        # Test invalid max_samples
        config = PipelineConfig(
            input_jsonl=Path("test.jsonl"),
            index_path=Path("index.txt"),
            stereo_dir=Path("stereo"),
            tmp_dir=Path("tmp"),
            user_spk_prompt=None,
            assistant_prompt=None,
            keep_temp=False,
            max_samples=0
        )
        config.input_jsonl.touch()
        with self.assertRaises(ValueError):
            config.validate()


class TestCommonPipelineLogic(unittest.TestCase):
    """Test the CommonPipelineLogic class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Mock(spec=WorkerConfig)
        self.setup = WorkerSetup(
            config=self.config,
            num_workers=2,
            task_queue=Mock(),
            result_queue=Mock(),
            worker_event=Mock(),
            planner_event=Mock(),
            completion_event=Mock()
        )
        self.logic = CommonPipelineLogic(self.setup)
    
    def test_common_pipeline_logic_creation(self):
        """Test CommonPipelineLogic can be created"""
        self.assertEqual(self.logic.setup, self.setup)
        self.assertEqual(self.logic.total_samples, 0)
        self.assertEqual(self.logic.total_duration, 0.0)
    
    def test_create_common_functions(self):
        """Test creation of common functions"""
        functions = self.logic.create_common_functions()
        
        # Should return flush_manifest, finalize_sample, enqueue_task functions
        self.assertIn('flush_manifest', functions)
        self.assertIn('finalize_sample', functions)
        self.assertIn('enqueue_task', functions)
    
    def test_get_final_stats(self):
        """Test final statistics calculation"""
        # Set some test data
        self.logic.total_samples = 10
        self.logic.total_duration = 25.5
        self.logic.start_time = time.time() - 100  # 100 seconds ago
        
        samples, duration = self.logic.get_final_stats()
        
        self.assertEqual(samples, 10)
        self.assertEqual(duration, 25.5)


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
        task = SynthesisTask(
            sample_id="test",
            text="test text",
            prompt_audio=Path("test.wav"),
            output_path=Path("output.wav")
        )
        
        result = _common_enqueue_task(task, mock_queue)
        
        self.assertTrue(result)
        mock_queue.put.assert_called_once_with(task)
    
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
        config.num_workers = 4
        config.batch_size = 32
        
        # Should not raise
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
    
    def test_unified_manager_with_common_logic(self):
        """Test UnifiedWorkerManager works with CommonPipelineLogic"""
        config = Mock(spec=WorkerConfig)
        setup = WorkerSetup(
            worker_cfg=config,
            worker_count=2,
            planner_buffer=10,
            manifest_buffer_size=100
        )
        
        manager = UnifiedWorkerManager(setup, use_processes=True)
        logic = CommonPipelineLogic(setup)
        
        # Both should be created successfully
        self.assertIsNotNone(manager)
        self.assertIsNotNone(logic)
        
        # Logic should have access to setup
        self.assertEqual(logic.setup, setup)


if __name__ == "__main__":
    unittest.main()
