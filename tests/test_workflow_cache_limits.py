"""
Tests for cache limit enforcement in automated workflow.

Verifies that cached outputs are revalidated when size limits change,
preventing oversized files from being treated as successful.
"""

import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

import pytest

from src.automated_workflow import AutomatedWorkflow
from src.config_manager import ConfigManager
from src.hardware_detector import HardwareDetector


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_base = tempfile.mkdtemp()
    input_dir = Path(temp_base) / 'input'
    output_dir = Path(temp_base) / 'output'
    temp_dir = Path(temp_base) / 'temp'
    
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    temp_dir.mkdir(parents=True)
    
    yield {
        'base': Path(temp_base),
        'input': input_dir,
        'output': output_dir,
        'temp': temp_dir
    }
    
    # Cleanup
    shutil.rmtree(temp_base, ignore_errors=True)


@pytest.fixture
def config_manager():
    """Create a config manager for testing."""
    return ConfigManager()


@pytest.fixture
def hardware():
    """Create a hardware detector for testing."""
    return HardwareDetector()


@pytest.fixture
def workflow(temp_dirs, config_manager, hardware):
    """Create an AutomatedWorkflow instance for testing."""
    video_compressor = MagicMock()
    gif_generator = MagicMock()
    
    workflow = AutomatedWorkflow(
        config_manager,
        hardware,
        video_compressor=video_compressor,
        gif_generator=gif_generator
    )
    
    workflow.input_dir = temp_dirs['input']
    workflow.output_dir = temp_dirs['output']
    workflow.temp_dir = temp_dirs['temp']
    workflow.use_cache = True
    
    return workflow


def create_test_gif(path: Path, size_mb: float):
    """Create a test GIF file with approximately the specified size."""
    # Create a minimal valid GIF header
    gif_header = b'GIF89a'
    # Calculate how many bytes we need (approximate)
    target_bytes = int(size_mb * 1024 * 1024)
    # Create padding to reach target size
    padding = b'\x00' * max(0, target_bytes - len(gif_header) - 1)
    gif_trailer = b';'
    
    with open(path, 'wb') as f:
        f.write(gif_header)
        f.write(padding)
        f.write(gif_trailer)


class TestCacheLimitEnforcement:
    """Test cache limit enforcement functionality."""
    
    def test_cache_hit_with_same_limit(self, workflow, temp_dirs):
        """Test that cache hit works when limit hasn't changed."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 5MB GIF
        create_test_gif(output_file, 5.0)
        
        # Record cache with 10MB limit
        workflow._active_cache_size_limit_mb = 10.0
        workflow._record_success_cache(input_file, 'single_gif', output_file, max_size_mb=10.0)
        
        # Check cache with same limit - should hit
        result = workflow._is_cached_success(input_file, max_size_mb=10.0)
        assert result is True, "Cache should hit when limit is the same"
    
    def test_cache_miss_when_limit_stricter(self, workflow, temp_dirs):
        """Test that cache misses when limit becomes stricter and file exceeds it."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 12MB GIF (exceeds 10MB limit)
        create_test_gif(output_file, 12.0)
        
        # Record cache with 15MB limit
        workflow._active_cache_size_limit_mb = 15.0
        workflow._record_success_cache(input_file, 'single_gif', output_file, max_size_mb=15.0)
        
        # Check cache with 10MB limit - should miss because file is too large
        result = workflow._is_cached_success(input_file, max_size_mb=10.0)
        assert result is False, "Cache should miss when file exceeds stricter limit"
        
        # Verify cache entry was removed
        key = workflow._make_cache_key(input_file)
        assert key not in workflow._cache_index, "Invalid cache entry should be removed"
    
    def test_cache_hit_when_limit_stricter_but_file_valid(self, workflow, temp_dirs):
        """Test that cache hits when limit becomes stricter but file still satisfies it."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 5MB GIF
        create_test_gif(output_file, 5.0)
        
        # Record cache with 10MB limit
        workflow._active_cache_size_limit_mb = 10.0
        workflow._record_success_cache(input_file, 'single_gif', output_file, max_size_mb=10.0)
        
        # Check cache with 8MB limit - should hit and update cache entry
        result = workflow._is_cached_success(input_file, max_size_mb=8.0)
        assert result is True, "Cache should hit when file satisfies stricter limit"
        
        # Verify cache entry was updated with stricter limit
        key = workflow._make_cache_key(input_file)
        assert key in workflow._cache_index, "Cache entry should still exist"
        assert workflow._cache_index[key]['max_size_mb'] == 8.0, "Cache entry should be updated with stricter limit"
    
    def test_cache_miss_for_legacy_entry_without_limit(self, workflow, temp_dirs):
        """Test that legacy cache entries without size limit are revalidated."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 12MB GIF
        create_test_gif(output_file, 12.0)
        
        # Manually create a legacy cache entry without max_size_mb
        key = workflow._make_cache_key(input_file)
        workflow._cache_index[key] = {
            'type': 'single_gif',
            'output': str(output_file),
            'input_signature': workflow._get_input_signature(input_file),
            'time': 1234567890.0,
            # No 'max_size_mb' field - legacy entry
        }
        
        # Check cache with 10MB limit - should miss because file exceeds limit
        result = workflow._is_cached_success(input_file, max_size_mb=10.0)
        assert result is False, "Legacy cache entry should be revalidated and fail if file exceeds limit"
        
        # Verify cache entry was removed
        assert key not in workflow._cache_index, "Invalid legacy cache entry should be removed"
    
    def test_cache_hit_for_legacy_entry_when_file_valid(self, workflow, temp_dirs):
        """Test that legacy cache entries are revalidated and updated when file is valid."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 5MB GIF
        create_test_gif(output_file, 5.0)
        
        # Manually create a legacy cache entry without max_size_mb
        key = workflow._make_cache_key(input_file)
        workflow._cache_index[key] = {
            'type': 'single_gif',
            'output': str(output_file),
            'input_signature': workflow._get_input_signature(input_file),
            'time': 1234567890.0,
            # No 'max_size_mb' field - legacy entry
        }
        
        # Check cache with 10MB limit - should hit and update cache entry
        result = workflow._is_cached_success(input_file, max_size_mb=10.0)
        assert result is True, "Legacy cache entry should be revalidated and pass if file is valid"
        
        # Verify cache entry was updated with limit
        assert key in workflow._cache_index, "Cache entry should still exist"
        assert workflow._cache_index[key].get('max_size_mb') == 10.0, "Cache entry should be updated with limit"
    
    def test_segments_cache_limit_enforcement(self, workflow, temp_dirs):
        """Test cache limit enforcement for segment folders."""
        input_file = temp_dirs['input'] / 'test.mp4'
        segments_dir = temp_dirs['output'] / 'test_segments'
        segments_dir.mkdir(parents=True)
        
        # Create two segment GIFs - one valid, one oversized
        create_test_gif(segments_dir / 'test_segment_01.gif', 5.0)
        create_test_gif(segments_dir / 'test_segment_02.gif', 12.0)  # Exceeds 10MB
        
        # Record cache with 15MB limit
        workflow._active_cache_size_limit_mb = 15.0
        workflow._record_success_cache(input_file, 'segments', segments_dir, max_size_mb=15.0)
        
        # Check cache with 10MB limit - should miss because one segment exceeds limit
        result = workflow._is_cached_success(input_file, max_size_mb=10.0)
        assert result is False, "Cache should miss when any segment exceeds stricter limit"
        
        # Verify cache entry was removed
        key = workflow._make_cache_key(input_file)
        assert key not in workflow._cache_index, "Invalid cache entry should be removed"
    
    def test_segments_cache_hit_when_all_valid(self, workflow, temp_dirs):
        """Test that segments cache hits when all segments satisfy stricter limit."""
        input_file = temp_dirs['input'] / 'test.mp4'
        segments_dir = temp_dirs['output'] / 'test_segments'
        segments_dir.mkdir(parents=True)
        
        # Create two valid segment GIFs
        create_test_gif(segments_dir / 'test_segment_01.gif', 4.0)
        create_test_gif(segments_dir / 'test_segment_02.gif', 5.0)
        
        # Record cache with 10MB limit
        workflow._active_cache_size_limit_mb = 10.0
        workflow._record_success_cache(input_file, 'segments', segments_dir, max_size_mb=10.0)
        
        # Check cache with 8MB limit - should hit and update cache entry
        result = workflow._is_cached_success(input_file, max_size_mb=8.0)
        assert result is True, "Cache should hit when all segments satisfy stricter limit"
        
        # Verify cache entry was updated
        key = workflow._make_cache_key(input_file)
        assert key in workflow._cache_index, "Cache entry should still exist"
        assert workflow._cache_index[key]['max_size_mb'] == 8.0, "Cache entry should be updated with stricter limit"
    
    def test_gif_input_cache_limit_enforcement(self, workflow, temp_dirs):
        """Test cache limit enforcement for GIF input files."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 12MB GIF
        create_test_gif(output_file, 12.0)
        
        # Record cache with 15MB limit
        workflow._active_cache_size_limit_mb = 15.0
        workflow._record_success_cache(input_file, 'gif_input', output_file, max_size_mb=15.0)
        
        # Check cache with 10MB limit - should miss because file exceeds limit
        result = workflow._is_cached_success(input_file, max_size_mb=10.0)
        assert result is False, "Cache should miss when GIF input exceeds stricter limit"
        
        # Verify cache entry was removed
        key = workflow._make_cache_key(input_file)
        assert key not in workflow._cache_index, "Invalid cache entry should be removed"
    
    def test_cache_persistence_with_size_limit(self, workflow, temp_dirs):
        """Test that cache entries with size limits are properly persisted."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 5MB GIF
        create_test_gif(output_file, 5.0)
        
        # Record cache with 10MB limit
        workflow._active_cache_size_limit_mb = 10.0
        workflow._record_success_cache(input_file, 'single_gif', output_file, max_size_mb=10.0)
        
        # Save cache index
        workflow._save_cache_index()
        
        # Reload cache index
        workflow._cache_index = {}
        workflow._load_cache_index()
        
        # Verify cache entry was loaded with size limit
        key = workflow._make_cache_key(input_file)
        assert key in workflow._cache_index, "Cache entry should be loaded"
        assert workflow._cache_index[key].get('max_size_mb') == 10.0, "Size limit should be preserved"
        
        # Verify cache still works after reload
        result = workflow._is_cached_success(input_file, max_size_mb=10.0)
        assert result is True, "Cache should work after reload"
    
    def test_cache_without_size_limit_uses_active_limit(self, workflow, temp_dirs):
        """Test that cache recording uses active workflow limit when not specified."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 5MB GIF
        create_test_gif(output_file, 5.0)
        
        # Set active limit
        workflow._active_cache_size_limit_mb = 12.0
        
        # Record cache without specifying limit
        workflow._record_success_cache(input_file, 'single_gif', output_file)
        
        # Verify cache entry has active limit
        key = workflow._make_cache_key(input_file)
        assert workflow._cache_index[key].get('max_size_mb') == 12.0, "Should use active workflow limit"
    
    def test_cache_validation_with_missing_file(self, workflow, temp_dirs):
        """Test that cache misses when output file doesn't exist."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Record cache entry
        workflow._active_cache_size_limit_mb = 10.0
        workflow._record_success_cache(input_file, 'single_gif', output_file, max_size_mb=10.0)
        
        # Don't create the output file - it should miss
        result = workflow._is_cached_success(input_file, max_size_mb=10.0)
        assert result is False, "Cache should miss when output file doesn't exist"
    
    def test_cache_validation_with_no_current_limit_but_cached_limit(self, workflow, temp_dirs):
        """Test that cache validates against cached limit when current limit is None."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 12MB GIF (exceeds 10MB cached limit)
        create_test_gif(output_file, 12.0)
        
        # Record cache with 10MB limit
        workflow._active_cache_size_limit_mb = 10.0
        workflow._record_success_cache(input_file, 'single_gif', output_file, max_size_mb=10.0)
        
        # Clear active limit to simulate workflow running without limit
        workflow._active_cache_size_limit_mb = None
        
        # Check cache without providing limit - should still validate against cached limit
        result = workflow._is_cached_success(input_file, max_size_mb=None)
        assert result is False, "Cache should miss when file exceeds cached limit even if current limit is None"
        
        # Verify cache entry was removed
        key = workflow._make_cache_key(input_file)
        assert key not in workflow._cache_index, "Invalid cache entry should be removed"
    
    def test_cache_hit_with_no_current_limit_but_file_valid(self, workflow, temp_dirs):
        """Test that cache hits when current limit is None but file satisfies cached limit."""
        input_file = temp_dirs['input'] / 'test.gif'
        output_file = temp_dirs['output'] / 'test.gif'
        
        # Create a 5MB GIF (satisfies 10MB cached limit)
        create_test_gif(output_file, 5.0)
        
        # Record cache with 10MB limit
        workflow._active_cache_size_limit_mb = 10.0
        workflow._record_success_cache(input_file, 'single_gif', output_file, max_size_mb=10.0)
        
        # Clear active limit to simulate workflow running without limit
        workflow._active_cache_size_limit_mb = None
        
        # Check cache without providing limit - should hit and validate against cached limit
        result = workflow._is_cached_success(input_file, max_size_mb=None)
        assert result is True, "Cache should hit when file satisfies cached limit even if current limit is None"
        
        # Verify cache entry still exists with original limit
        key = workflow._make_cache_key(input_file)
        assert key in workflow._cache_index, "Cache entry should still exist"
        assert workflow._cache_index[key].get('max_size_mb') == 10.0, "Cached limit should be preserved"

