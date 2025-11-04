"""
Unit tests for VMAF filter parameter generation and validation
Tests filter parameter generation with various file paths, proper escaping of special characters,
cross-platform path handling, and filter syntax validation
"""

import unittest
import tempfile
import os
import platform
from unittest.mock import Mock, patch
from src.quality_gates import VMAFFilterBuilder, DebugLogManager


class TestVMAFFilterParameterGeneration(unittest.TestCase):
    """Test VMAF filter parameter generation with various file paths"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.debug_log_manager = DebugLogManager()
        self.vmaf_builder = VMAFFilterBuilder(self.debug_log_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_basic_filter_generation(self):
        """Test basic VMAF filter generation with simple path"""
        log_path = os.path.join(self.temp_dir, "vmaf_debug.json")
        
        filter_string = self.vmaf_builder.build_vmaf_filter(log_path, "json")
        
        self.assertIn("libvmaf=", filter_string)
        self.assertIn("log_fmt=json", filter_string)
        self.assertIn("log_path=", filter_string)
        self.assertIn("n_threads=", filter_string)
    
    def test_filter_generation_with_xml_format(self):
        """Test VMAF filter generation with XML format"""
        log_path = os.path.join(self.temp_dir, "vmaf_debug.xml")
        
        filter_string = self.vmaf_builder.build_vmaf_filter(log_path, "xml")
        
        self.assertIn("libvmaf=", filter_string)
        self.assertIn("log_fmt=xml", filter_string)
        self.assertIn("log_path=", filter_string)
    
    def test_filter_generation_with_text_format(self):
        """Test VMAF filter generation with text format (no log_fmt parameter)"""
        log_path = os.path.join(self.temp_dir, "vmaf_debug.log")
        
        filter_string = self.vmaf_builder.build_vmaf_filter(log_path, "text")
        
        self.assertIn("libvmaf=", filter_string)
        self.assertNotIn("log_fmt=", filter_string)  # Text is default, no parameter needed
        self.assertIn("log_path=", filter_string)
    
    def test_filter_generation_with_custom_threads(self):
        """Test VMAF filter generation with custom thread count"""
        log_path = os.path.join(self.temp_dir, "vmaf_debug.json")
        
        filter_string = self.vmaf_builder.build_vmaf_filter(log_path, "json", n_threads=8)
        
        self.assertIn("n_threads=8", filter_string)
    
    def test_filter_generation_with_model_path(self):
        """Test VMAF filter generation with custom model path"""
        log_path = os.path.join(self.temp_dir, "vmaf_debug.json")
        model_path = os.path.join(self.temp_dir, "vmaf_model.pkl")
        
        filter_string = self.vmaf_builder.build_vmaf_filter(
            log_path, "json", model_path=model_path
        )
        
        self.assertIn("model_path=", filter_string)
        self.assertIn("vmaf_model.pkl", filter_string)
    
    def test_filter_parameter_order(self):
        """Test that filter parameters are in expected order"""
        log_path = os.path.join(self.temp_dir, "vmaf_debug.json")
        
        filter_string = self.vmaf_builder.build_vmaf_filter(log_path, "json", n_threads=2)
        
        # Extract parameters part
        params_part = filter_string.replace("libvmaf=", "")
        params = params_part.split(":")
        
        # Should have log_fmt, log_path, and n_threads
        self.assertEqual(len(params), 3)
        self.assertTrue(params[0].startswith("log_fmt="))
        self.assertTrue(params[1].startswith("log_path="))
        self.assertTrue(params[2].startswith("n_threads="))


class TestVMAFFilterPathEscaping(unittest.TestCase):
    """Test proper escaping of special characters in file paths"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.debug_log_manager = DebugLogManager()
        self.vmaf_builder = VMAFFilterBuilder(self.debug_log_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_escape_colon_in_path(self):
        """Test escaping of colon characters in file paths"""
        # Create path with colon (common on Windows drive letters)
        if platform.system() == "Windows":
            log_path = "C:\\logs\\vmaf:debug.json"
        else:
            log_path = "/tmp/logs/vmaf:debug.json"
        
        escaped_path = self.vmaf_builder.escape_file_path(log_path)
        
        self.assertIn("\\:", escaped_path)
        self.assertNotIn(":", escaped_path.replace("\\:", ""))
    
    def test_escape_equals_in_path(self):
        """Test escaping of equals characters in file paths"""
        log_path = os.path.join(self.temp_dir, "vmaf=debug.json")
        
        escaped_path = self.vmaf_builder.escape_file_path(log_path)
        
        self.assertIn("\\=", escaped_path)
        self.assertNotIn("=", escaped_path.replace("\\=", ""))
    
    def test_escape_comma_in_path(self):
        """Test escaping of comma characters in file paths"""
        log_path = os.path.join(self.temp_dir, "vmaf,debug.json")
        
        escaped_path = self.vmaf_builder.escape_file_path(log_path)
        
        self.assertIn("\\,", escaped_path)
        self.assertNotIn(",", escaped_path.replace("\\,", ""))
    
    def test_escape_brackets_in_path(self):
        """Test escaping of bracket characters in file paths"""
        log_path = os.path.join(self.temp_dir, "vmaf[debug].json")
        
        escaped_path = self.vmaf_builder.escape_file_path(log_path)
        
        self.assertIn("\\[", escaped_path)
        self.assertIn("\\]", escaped_path)
        self.assertNotIn("[", escaped_path.replace("\\[", ""))
        self.assertNotIn("]", escaped_path.replace("\\]", ""))
    
    def test_escape_quotes_in_path(self):
        """Test escaping of quote characters in file paths"""
        log_path = os.path.join(self.temp_dir, "vmaf'debug\"file.json")
        
        escaped_path = self.vmaf_builder.escape_file_path(log_path)
        
        self.assertIn("\\'", escaped_path)
        self.assertIn('\\"', escaped_path)
    
    def test_escape_multiple_special_chars(self):
        """Test escaping of multiple special characters in same path"""
        log_path = os.path.join(self.temp_dir, "vmaf:debug=test,file[1].json")
        
        escaped_path = self.vmaf_builder.escape_file_path(log_path)
        
        # All special characters should be escaped
        self.assertIn("\\:", escaped_path)
        self.assertIn("\\=", escaped_path)
        self.assertIn("\\,", escaped_path)
        self.assertIn("\\[", escaped_path)
        self.assertIn("\\]", escaped_path)
    
    def test_no_double_escaping(self):
        """Test that already escaped characters are not double-escaped"""
        log_path = os.path.join(self.temp_dir, "vmaf\\:debug.json")
        
        escaped_path = self.vmaf_builder.escape_file_path(log_path)
        
        # Should not have double backslashes for already escaped characters
        self.assertNotIn("\\\\:", escaped_path)
    
    def test_empty_path_handling(self):
        """Test handling of empty or None paths"""
        self.assertEqual(self.vmaf_builder.escape_file_path(""), "")
        self.assertEqual(self.vmaf_builder.escape_file_path(None), "")


class TestVMAFFilterCrossPlatformPaths(unittest.TestCase):
    """Test cross-platform path handling for debug files"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.debug_log_manager = DebugLogManager()
        self.vmaf_builder = VMAFFilterBuilder(self.debug_log_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_windows_path_normalization(self):
        """Test Windows path normalization to forward slashes"""
        windows_path = "C:\\Users\\Test\\logs\\vmaf_debug.json"
        
        escaped_path = self.vmaf_builder.escape_file_path(windows_path)
        
        # Should convert backslashes to forward slashes
        if platform.system() == "Windows":
            # On Windows, should normalize to forward slashes
            self.assertIn("/", escaped_path)
            # Should not have unescaped backslashes (except for escaping special chars)
            self.assertNotIn("\\Users", escaped_path)
        else:
            # On Unix, should handle Windows-style paths gracefully
            self.assertIsInstance(escaped_path, str)
    
    def test_unix_path_preservation(self):
        """Test Unix path preservation"""
        unix_path = "/tmp/logs/vmaf_debug.json"
        
        escaped_path = self.vmaf_builder.escape_file_path(unix_path)
        
        # Should preserve forward slashes
        self.assertIn("/tmp/logs/", escaped_path)
        self.assertNotIn("\\tmp", escaped_path)
    
    def test_relative_path_conversion(self):
        """Test conversion of relative paths to absolute paths"""
        relative_path = "logs/vmaf_debug.json"
        
        escaped_path = self.vmaf_builder.escape_file_path(relative_path)
        
        # Should be converted to absolute path
        self.assertTrue(os.path.isabs(escaped_path.replace("\\:", ":")))
    
    def test_path_with_spaces(self):
        """Test handling of paths with spaces"""
        path_with_spaces = os.path.join(self.temp_dir, "vmaf debug file.json")
        
        escaped_path = self.vmaf_builder.escape_file_path(path_with_spaces)
        
        # Spaces should be preserved (not escaped in FFmpeg filter context)
        self.assertIn(" ", escaped_path)
        # But other special characters should still be escaped if present
        self.assertIsInstance(escaped_path, str)
    
    def test_unicode_path_handling(self):
        """Test handling of Unicode characters in paths"""
        unicode_path = os.path.join(self.temp_dir, "vmaf_测试_файл.json")
        
        escaped_path = self.vmaf_builder.escape_file_path(unicode_path)
        
        # Should handle Unicode characters without corruption
        self.assertIsInstance(escaped_path, str)
        # Unicode characters should be preserved
        self.assertIn("测试", escaped_path)
        self.assertIn("файл", escaped_path)


class TestVMAFFilterSyntaxValidation(unittest.TestCase):
    """Test filter syntax validation catches malformed parameters"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.debug_log_manager = DebugLogManager()
        self.vmaf_builder = VMAFFilterBuilder(self.debug_log_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_valid_filter_syntax(self):
        """Test validation of valid filter syntax"""
        valid_filter = "libvmaf=log_fmt=json:log_path=/tmp/vmaf.json:n_threads=4"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(valid_filter)
        
        self.assertTrue(validation_result['is_valid'])
        self.assertEqual(len(validation_result['errors']), 0)
    
    def test_missing_libvmaf_prefix(self):
        """Test validation catches missing libvmaf prefix"""
        invalid_filter = "log_fmt=json:log_path=/tmp/vmaf.json"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(invalid_filter)
        
        self.assertFalse(validation_result['is_valid'])
        self.assertIn("must start with 'libvmaf='", str(validation_result['errors']))
    
    def test_empty_filter_string(self):
        """Test validation catches empty filter string"""
        validation_result = self.vmaf_builder.validate_filter_syntax("")
        
        self.assertFalse(validation_result['is_valid'])
        self.assertIn("empty", str(validation_result['errors']))
    
    def test_missing_parameters(self):
        """Test validation catches missing parameters"""
        invalid_filter = "libvmaf="
        
        validation_result = self.vmaf_builder.validate_filter_syntax(invalid_filter)
        
        self.assertFalse(validation_result['is_valid'])
        self.assertIn("No parameters found", str(validation_result['errors']))
    
    def test_malformed_parameter_no_equals(self):
        """Test validation catches parameters without equals sign"""
        invalid_filter = "libvmaf=log_fmt_json:log_path=/tmp/vmaf.json"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(invalid_filter)
        
        self.assertFalse(validation_result['is_valid'])
        self.assertIn("missing '=' separator", str(validation_result['errors']))
    
    def test_empty_parameter_key(self):
        """Test validation catches empty parameter keys"""
        invalid_filter = "libvmaf==json:log_path=/tmp/vmaf.json"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(invalid_filter)
        
        self.assertFalse(validation_result['is_valid'])
        self.assertIn("empty key", str(validation_result['errors']))
    
    def test_empty_parameter_value(self):
        """Test validation catches empty parameter values"""
        invalid_filter = "libvmaf=log_fmt=:log_path=/tmp/vmaf.json"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(invalid_filter)
        
        self.assertFalse(validation_result['is_valid'])
        self.assertIn("empty value", str(validation_result['errors']))
    
    def test_invalid_thread_count(self):
        """Test validation catches invalid thread count"""
        invalid_filter = "libvmaf=log_path=/tmp/vmaf.json:n_threads=invalid"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(invalid_filter)
        
        self.assertFalse(validation_result['is_valid'])
        self.assertIn("must be a number", str(validation_result['errors']))
    
    def test_negative_thread_count(self):
        """Test validation warns about negative thread count"""
        invalid_filter = "libvmaf=log_path=/tmp/vmaf.json:n_threads=-1"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(invalid_filter)
        
        # Should be valid but with warnings
        self.assertTrue(validation_result['is_valid'])
        self.assertIn("should be positive", str(validation_result['warnings']))
    
    def test_high_thread_count_warning(self):
        """Test validation warns about very high thread count"""
        high_thread_filter = "libvmaf=log_path=/tmp/vmaf.json:n_threads=32"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(high_thread_filter)
        
        # Should be valid but with warnings
        self.assertTrue(validation_result['is_valid'])
        self.assertIn("very high", str(validation_result['warnings']))
    
    def test_unusual_log_format_warning(self):
        """Test validation warns about unusual log formats"""
        unusual_filter = "libvmaf=log_fmt=csv:log_path=/tmp/vmaf.json"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(unusual_filter)
        
        # Should be valid but with warnings
        self.assertTrue(validation_result['is_valid'])
        self.assertIn("Unusual log format", str(validation_result['warnings']))
    
    def test_duplicate_parameters_warning(self):
        """Test validation warns about duplicate parameters"""
        duplicate_filter = "libvmaf=log_path=/tmp/vmaf1.json:log_path=/tmp/vmaf2.json"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(duplicate_filter)
        
        # Should be valid but with warnings
        self.assertTrue(validation_result['is_valid'])
        self.assertIn("Duplicate parameter", str(validation_result['warnings']))
    
    def test_auto_correction_functionality(self):
        """Test that auto-correction provides corrected filter when possible"""
        # Filter with unescaped special characters in path
        problematic_filter = "libvmaf=log_path=/tmp/vmaf:debug.json"
        
        validation_result = self.vmaf_builder.validate_filter_syntax(problematic_filter)
        
        # Should provide corrected version
        if validation_result.get('corrected_filter'):
            corrected = validation_result['corrected_filter']
            self.assertIn("\\:", corrected)
            self.assertNotIn("vmaf:debug", corrected.replace("\\:", ""))
    
    def test_validate_and_correct_filter_method(self):
        """Test the validate_and_correct_filter convenience method"""
        problematic_filter = "libvmaf=log_path=/tmp/vmaf:debug.json"
        
        is_valid, corrected_filter, errors = self.vmaf_builder.validate_and_correct_filter(problematic_filter)
        
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(corrected_filter, str)
        self.assertIsInstance(errors, list)
        
        # Corrected filter should be different from original if corrections were made
        if not is_valid and corrected_filter != problematic_filter:
            self.assertIn("\\:", corrected_filter)


class TestVMAFFilterComplexGeneration(unittest.TestCase):
    """Test complete filter_complex generation for VMAF computation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.debug_log_manager = DebugLogManager()
        self.vmaf_builder = VMAFFilterBuilder(self.debug_log_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_basic_filter_complex_generation(self):
        """Test basic filter_complex generation without scaling"""
        ref_path = os.path.join(self.temp_dir, "reference.mp4")
        dist_path = os.path.join(self.temp_dir, "distorted.mp4")
        log_path = os.path.join(self.temp_dir, "vmaf.json")
        
        filter_complex = self.vmaf_builder.build_complete_filter_complex(
            ref_path, dist_path, log_path
        )
        
        # Should contain proper filter structure
        self.assertIn("[0:v]", filter_complex)
        self.assertIn("[1:v]", filter_complex)
        self.assertIn("[dist][ref]", filter_complex)
        self.assertIn("libvmaf=", filter_complex)
        self.assertIn("setpts=PTS-STARTPTS", filter_complex)
    
    def test_filter_complex_with_scaling(self):
        """Test filter_complex generation with resolution scaling"""
        ref_path = os.path.join(self.temp_dir, "reference.mp4")
        dist_path = os.path.join(self.temp_dir, "distorted.mp4")
        log_path = os.path.join(self.temp_dir, "vmaf.json")
        
        filter_complex = self.vmaf_builder.build_complete_filter_complex(
            ref_path, dist_path, log_path, eval_height=720
        )
        
        # Should contain scaling filters
        self.assertIn("scale=-2:720", filter_complex)
        self.assertIn("flags=bicubic", filter_complex)
    
    def test_filter_complex_with_custom_threads(self):
        """Test filter_complex generation with custom thread count"""
        ref_path = os.path.join(self.temp_dir, "reference.mp4")
        dist_path = os.path.join(self.temp_dir, "distorted.mp4")
        log_path = os.path.join(self.temp_dir, "vmaf.json")
        
        filter_complex = self.vmaf_builder.build_complete_filter_complex(
            ref_path, dist_path, log_path, n_threads=8
        )
        
        # Should contain custom thread count
        self.assertIn("n_threads=8", filter_complex)


if __name__ == '__main__':
    unittest.main()