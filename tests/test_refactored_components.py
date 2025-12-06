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
        # Mock behaviors that might be called during init
        self.config.mock_inference = True
        
        self.setup = WorkerSetup(
            worker_cfg=self.config,
            worker_count=2,
            planner_buffer=10,
            manifest_buffer_size=100
        )
    
    @patch('multiprocessing.get_context')
    def test_process_worker_manager_creation(self, mock_get_context):
        """Test UnifiedWorkerManager with process workers"""
        # Mock context and process to avoid actual spawning
        mock_ctx = Mock()
        mock_get_context.return_value = mock_ctx
        
        manager = UnifiedWorkerManager(self.setup, use_processes=True)
        self.assertTrue(manager.use_processes)
        self.assertEqual(manager.setup, self.setup)
    
    def test_thread_worker_manager_creation(self):
        """Test UnifiedWorkerManager with thread workers"""
        # We need to mock _create_tts since it's called in __init__ for threads
        with patch('tools.build_moshi_dataset_with_indexts._create_tts') as mock_create:
            manager = UnifiedWorkerManager(self.setup, use_processes=False)
            self.assertFalse(manager.use_processes)
            self.assertEqual(manager.setup, self.setup)
    
    @patch('multiprocessing.get_context')
    def test_setup_process_workers(self, mock_get_context):
        """Test process worker setup"""
        mock_ctx = Mock()
        mock_get_context.return_value = mock_ctx
        mock_process = Mock()
        mock_ctx.Process.return_value = mock_process
        
        manager = UnifiedWorkerManager(self.setup, use_processes=True)
        
        # Should create num_workers processes
        self.assertEqual(mock_ctx.Process.call_count, 2)
        self.assertEqual(len(manager.processes), 2)
        # Verify start was called
        self.assertEqual(mock_process.start.call_count, 2)
    
    @patch('tools.build_moshi_dataset_with_indexts._create_tts')
    def test_setup_thread_workers(self, mock_create):
        """Test thread worker setup"""
        # For thread workers, we don't spawn threads in init anymore, just setup queues and tts/locks
        manager = UnifiedWorkerManager(self.setup, use_processes=False)
        
        # Verify TTS was created
        mock_create.assert_called_once()
        self.assertIsNotNone(manager.task_queue)
        self.assertIsNotNone(manager.result_queue)


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
        with patch.object(Path, 'exists', return_value=True):
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
        with patch.object(Path, 'exists', return_value=True):
            with self.assertRaises(ValueError):
                config.validate()


class TestCommonPipelineLogic(unittest.TestCase):
    """Test the CommonPipelineLogic class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.worker_config = Mock(spec=WorkerConfig)
        self.worker_setup = WorkerSetup(
            worker_cfg=self.worker_config,
            worker_count=2,
            planner_buffer=10,
            manifest_buffer_size=100
        )
        self.pipeline_config = Mock(spec=PipelineConfig)
        self.pipeline_config.stereo_dir = Path("stereo")
        
        self.manager = Mock(spec=UnifiedWorkerManager)
        self.manager.get_task_queue.return_value = Mock()
        self.manager.stop_event = Mock()
        
        self.logic = CommonPipelineLogic(
            config=self.pipeline_config,
            worker_setup=self.worker_setup,
            manager=self.manager
        )
    
    def test_common_pipeline_logic_creation(self):
        """Test CommonPipelineLogic can be created"""
        self.assertEqual(self.logic.worker_setup, self.worker_setup)
        self.assertEqual(self.logic.total_samples, 0)
        self.assertEqual(self.logic.total_duration, 0.0)
    
    def test_create_common_functions(self):
        """Test creation of common functions"""
        functions = self.logic.create_common_functions()
        
        # Should return flush_manifest, finalize_sample, enqueue_task functions
        self.assertEqual(len(functions), 3)
        self.assertTrue(callable(functions[0]))
        self.assertTrue(callable(functions[1]))
        self.assertTrue(callable(functions[2]))
    
    def test_get_final_stats(self):
        """Test final statistics calculation"""
        # Set some test data
        self.logic.total_samples = 10
        self.logic.total_duration = 25.5
        self.logic.start_time = time.perf_counter() - 100  # 100 seconds ago
        
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
    
    def test_validate_worker_config(self):
        """Test _validate_worker_config function"""
        # Test valid config
        config = Mock(spec=WorkerConfig)
        config.mock_inference = True
        
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
        # This test remains similar, checking they can coexist
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
    
    @patch('multiprocessing.get_context')
    def test_unified_manager_with_common_logic(self, mock_get_context):
        """Test UnifiedWorkerManager works with CommonPipelineLogic"""
        mock_ctx = Mock()
        mock_get_context.return_value = mock_ctx
        
        worker_config = Mock(spec=WorkerConfig)
        worker_config.mock_inference = True
        
        setup = WorkerSetup(
            worker_cfg=worker_config,
            worker_count=2,
            planner_buffer=10,
            manifest_buffer_size=100
        )
        
        manager = UnifiedWorkerManager(setup, use_processes=True)
        
        pipeline_config = Mock(spec=PipelineConfig)
        
        logic = CommonPipelineLogic(pipeline_config, setup, manager)
        
        # Both should be created successfully
        self.assertIsNotNone(manager)
        self.assertIsNotNone(logic)
        self.assertEqual(logic.manager, manager)


if __name__ == "__main__":
    unittest.main()
