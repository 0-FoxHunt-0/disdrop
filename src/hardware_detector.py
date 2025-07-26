"""
Hardware Detection for Video Compressor
Detects available GPU hardware acceleration support for NVIDIA and AMD
"""

import subprocess
import psutil
import platform
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class GPUVendor(Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"

class HardwareDetector:
    def __init__(self):
        self.system_info = self._get_system_info()
        self.gpu_info = self._detect_gpus()
        self.ffmpeg_encoders = self._detect_ffmpeg_encoders()
        self.validation_results = {}  # Track validation results for encoders
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information"""
        return {
            'platform': platform.system(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
    
    def _detect_gpus(self) -> List[Dict[str, str]]:
        """Detect available GPUs and their vendors"""
        gpus = []
        
        try:
            # Try to detect NVIDIA GPUs using nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpus.append({
                                'vendor': GPUVendor.NVIDIA.value,
                                'name': parts[0].strip(),
                                'memory_mb': parts[1].strip(),
                                'driver_available': True
                            })
                logger.info(f"Detected {len(gpus)} NVIDIA GPU(s)")
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("NVIDIA drivers not available or nvidia-smi not found")
        
        # Try to detect AMD GPUs with multiple methods
        amd_gpus_found = 0
        try:
            if platform.system() == "Windows":
                # Method 1: Try PowerShell first (more reliable on modern Windows)
                try:
                    ps_cmd = 'Get-WmiObject -Class Win32_VideoController | Select-Object Name, AdapterRAM | Where-Object {$_.Name -match "AMD|ATI|Radeon"} | ForEach-Object {"$($_.Name)|$($_.AdapterRAM)"}'
                    result = subprocess.run(['powershell', '-Command', ps_cmd], 
                                          capture_output=True, text=True, timeout=15)
                    if result.returncode == 0 and result.stdout.strip():
                        for line in result.stdout.strip().split('\n'):
                            if '|' in line:
                                parts = line.split('|')
                                name = parts[0].strip()
                                memory = parts[1].strip() if len(parts) > 1 else 'Unknown'
                                if name and (name.upper().find('AMD') != -1 or 
                                           name.upper().find('ATI') != -1 or 
                                           name.upper().find('RADEON') != -1):
                                    # Convert bytes to MB if it's a number
                                    memory_mb = 'Unknown'
                                    try:
                                        if memory.isdigit() and int(memory) > 0:
                                            memory_mb = str(int(memory) // (1024 * 1024))
                                    except:
                                        pass
                                    
                                    gpus.append({
                                        'vendor': GPUVendor.AMD.value,
                                        'name': name,
                                        'memory_mb': memory_mb,
                                        'driver_available': True
                                    })
                                    amd_gpus_found += 1
                        
                        if amd_gpus_found > 0:
                            logger.info(f"Detected {amd_gpus_found} AMD GPU(s) via PowerShell")
                
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    logger.debug("PowerShell AMD detection failed, trying wmic")
                
                # Method 2: Fallback to wmic if PowerShell failed and no AMD GPUs found
                if amd_gpus_found == 0:
                    try:
                        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            for line in result.stdout.split('\n'):
                                line = line.strip()
                                if line and (line.upper().find('AMD') != -1 or 
                                           line.upper().find('ATI') != -1 or 
                                           line.upper().find('RADEON') != -1):
                                    gpus.append({
                                        'vendor': GPUVendor.AMD.value,
                                        'name': line,
                                        'memory_mb': 'Unknown',
                                        'driver_available': True
                                    })
                                    amd_gpus_found += 1
                            
                            if amd_gpus_found > 0:
                                logger.info(f"Detected {amd_gpus_found} AMD GPU(s) via wmic")
                    
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                        logger.debug("wmic AMD detection also failed")
                
                # Method 3: Registry-based detection as final fallback
                if amd_gpus_found == 0:
                    try:
                        reg_cmd = 'Get-ItemProperty "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\*" | Where-Object {$_.DriverDesc -match "AMD|ATI|Radeon"} | Select-Object DriverDesc'
                        result = subprocess.run(['powershell', '-Command', reg_cmd], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0 and result.stdout.strip():
                            for line in result.stdout.split('\n'):
                                line = line.strip()
                                if line and not line.startswith('DriverDesc') and not line.startswith('----'):
                                    if (line.upper().find('AMD') != -1 or 
                                        line.upper().find('ATI') != -1 or 
                                        line.upper().find('RADEON') != -1):
                                        gpus.append({
                                            'vendor': GPUVendor.AMD.value,
                                            'name': line,
                                            'memory_mb': 'Unknown',
                                            'driver_available': True
                                        })
                                        amd_gpus_found += 1
                            
                            if amd_gpus_found > 0:
                                logger.info(f"Detected {amd_gpus_found} AMD GPU(s) via registry")
                    
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                        logger.debug("Registry-based AMD detection failed")
            
            elif platform.system() == "Linux":
                # On Linux, try to use lspci to detect AMD GPUs
                result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'VGA' in line and ('AMD' in line.upper() or 'ATI' in line.upper()):
                            gpus.append({
                                'vendor': GPUVendor.AMD.value,
                                'name': line.split(': ')[1] if ': ' in line else line,
                                'memory_mb': 'Unknown',
                                'driver_available': True
                            })
                            amd_gpus_found += 1
                    
                    if amd_gpus_found > 0:
                        logger.info(f"Detected {amd_gpus_found} AMD GPU(s) via lspci")
                            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Could not detect AMD GPUs using any method")
        
        # Log final GPU detection results
        if not gpus:
            logger.warning("No GPUs detected - hardware acceleration may not be available")
        else:
            total_gpus = len(gpus)
            nvidia_count = sum(1 for gpu in gpus if gpu['vendor'] == GPUVendor.NVIDIA.value)
            amd_count = sum(1 for gpu in gpus if gpu['vendor'] == GPUVendor.AMD.value)
            logger.info(f"Total GPUs detected: {total_gpus} (NVIDIA: {nvidia_count}, AMD: {amd_count})")
        
        return gpus
    
    def _detect_ffmpeg_encoders(self) -> Dict[str, bool]:
        """Detect available FFmpeg hardware encoders"""
        encoders = {
            'h264_nvenc': False,  # NVIDIA H.264
            'hevc_nvenc': False,  # NVIDIA H.265
            'h264_amf': False,    # AMD H.264
            'hevc_amf': False,    # AMD H.265
            'av1_amf': False,     # AMD AV1
            'h264_qsv': False,    # Intel QuickSync H.264
            'hevc_qsv': False,    # Intel QuickSync H.265
            'libx264': False,     # Software H.264
            'libx265': False      # Software H.265
        }
        
        try:
            # Check FFmpeg encoders
            result = subprocess.run(['ffmpeg', '-encoders'], 
                                  capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                output = result.stdout + result.stderr
                
                for encoder in encoders.keys():
                    if encoder in output:
                        encoders[encoder] = True
                        logger.debug(f"FFmpeg encoder available: {encoder}")
                
                logger.info(f"Detected {sum(encoders.values())} available FFmpeg encoders")
                
                # For AMD AMF encoders, perform additional validation
                if encoders.get('h264_amf') or encoders.get('hevc_amf'):
                    self._validate_amd_amf_support(encoders)
                
                # For NVIDIA NVENC encoders, perform basic validation
                if encoders.get('h264_nvenc') or encoders.get('hevc_nvenc'):
                    self._validate_nvidia_nvenc_support(encoders)
                    
            else:
                logger.warning("FFmpeg not found or not working properly")
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg not available - hardware acceleration will not work")
        
        return encoders
    
    def _validate_amd_amf_support(self, encoders: Dict[str, bool]) -> None:
        """Validate AMD AMF encoder support with comprehensive testing"""
        if not (encoders.get('h264_amf') or encoders.get('hevc_amf')):
            return
            
        logger.info("Validating AMD AMF encoder functionality...")
        
        # Ensure validation_results is initialized
        if not hasattr(self, 'validation_results'):
            self.validation_results = {}
        
        # Test with ultra-conservative settings to match our encoding parameters
        test_commands = [
            # Test h264_amf with our conservative settings
            [
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1',
                '-c:v', 'h264_amf', 
                '-usage', 'transcoding', '-quality', 'speed', '-rc', 'cbr',
                '-b:v', '500k', '-maxrate', '510k', '-bufsize', '550k',
                '-profile:v', 'baseline', '-level', '3.1', '-refs', '1',
                '-qp', '25', '-frames:v', '5', '-f', 'null', '-'
            ]
        ]
        
        validation_results = []
        
        for i, test_cmd in enumerate(test_commands):
            try:
                logger.debug(f"Running AMD AMF validation test {i+1}...")
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0:
                    validation_results.append(True)
                    logger.debug(f"AMD AMF validation test {i+1} passed")
                else:
                    validation_results.append(False)
                    error_output = result.stderr.lower()
                    
                    # Analyze specific error patterns
                    if 'invalid argument' in error_output:
                        logger.warning("AMD AMF validation failed: Invalid argument error detected - encoder may have compatibility issues")
                    elif 'failed to initialize' in error_output or 'amf' in error_output:
                        logger.warning("AMD AMF validation failed: Encoder initialization failed - may need driver update")
                    elif 'no device' in error_output:
                        logger.warning("AMD AMF validation failed: No compatible device found")
                    else:
                        logger.warning(f"AMD AMF validation failed with unknown error: {result.stderr[:200]}")
                    
                    logger.debug(f"Full AMD AMF validation error: {result.stderr}")
                        
            except subprocess.TimeoutExpired:
                validation_results.append(False)
                logger.warning("AMD AMF validation test timed out - encoder may be unstable")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                validation_results.append(False)
                logger.warning(f"AMD AMF validation test failed: {e}")
        
        # Report overall validation results and disable failed encoders
        if any(validation_results):
            successful_tests = sum(validation_results)
            total_tests = len(validation_results)
            if successful_tests == total_tests:
                logger.info("AMD AMF hardware acceleration validated successfully")
                # Mark AMD AMF encoders as validated
                if encoders.get('h264_amf'):
                    self.validation_results['h264_amf'] = True
                if encoders.get('hevc_amf'):
                    self.validation_results['hevc_amf'] = True
            else:
                logger.warning(f"AMD AMF validation partially successful ({successful_tests}/{total_tests} tests passed)")
                logger.info("AMD AMF encoder will be used but may fallback to software encoding on errors")
                # Mark as partially validated
                if encoders.get('h264_amf'):
                    self.validation_results['h264_amf'] = 'partial'
                if encoders.get('hevc_amf'):
                    self.validation_results['hevc_amf'] = 'partial'
        else:
            logger.warning("All AMD AMF validation tests failed - encoder detected but may not function properly")
            logger.info("Software fallback will be used automatically if AMD AMF fails during encoding")
            # Disable AMD AMF encoders that failed validation
            if encoders.get('h264_amf'):
                encoders['h264_amf'] = False
                self.validation_results['h264_amf'] = False
                logger.info("Disabled h264_amf encoder due to validation failure")
            if encoders.get('hevc_amf'):
                encoders['hevc_amf'] = False
                self.validation_results['hevc_amf'] = False
                logger.info("Disabled hevc_amf encoder due to validation failure")
    
    def _validate_nvidia_nvenc_support(self, encoders: Dict[str, bool]) -> None:
        """Validate NVIDIA NVENC encoder support with comprehensive testing"""
        if not (encoders.get('h264_nvenc') or encoders.get('hevc_nvenc')):
            return
            
        logger.info("Validating NVIDIA NVENC encoder functionality...")
        
        # Ensure validation_results is initialized
        if not hasattr(self, 'validation_results'):
            self.validation_results = {}
        
        # Test with ultra-conservative settings to match our encoding parameters
        test_commands = [
            # Test h264_nvenc with our conservative settings
            [
                'ffmpeg', '-f', 'lavfi', '-i', 'testsrc=duration=1:size=320x240:rate=1',
                '-c:v', 'h264_nvenc', 
                '-usage', 'transcoding', '-quality', 'speed', '-rc', 'cbr',
                '-b:v', '500k', '-maxrate', '510k', '-bufsize', '550k',
                '-profile:v', 'baseline', '-level', '3.1', '-refs', '1',
                '-qp', '25', '-frames:v', '5', '-f', 'null', '-'
            ]
        ]
        
        validation_results = []
        
        for i, test_cmd in enumerate(test_commands):
            try:
                logger.debug(f"Running NVIDIA NVENC validation test {i+1}...")
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0:
                    validation_results.append(True)
                    logger.debug(f"NVIDIA NVENC validation test {i+1} passed")
                else:
                    validation_results.append(False)
                    error_output = result.stderr.lower()
                    
                    # Analyze specific error patterns
                    if 'invalid argument' in error_output:
                        logger.warning("NVIDIA NVENC validation failed: Invalid argument error detected - encoder may have compatibility issues")
                    elif 'failed to initialize' in error_output or 'nvenc' in error_output:
                        logger.warning("NVIDIA NVENC validation failed: Encoder initialization failed - may need driver update")
                    elif 'no device' in error_output:
                        logger.warning("NVIDIA NVENC validation failed: No compatible device found")
                    else:
                        logger.warning(f"NVIDIA NVENC validation failed with unknown error: {result.stderr[:200]}")
                    
                    logger.debug(f"Full NVIDIA NVENC validation error: {result.stderr}")
                        
            except subprocess.TimeoutExpired:
                validation_results.append(False)
                logger.warning("NVIDIA NVENC validation test timed out - encoder may be unstable")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                validation_results.append(False)
                logger.warning(f"NVIDIA NVENC validation test failed: {e}")
        
        # Report overall validation results and disable failed encoders
        if any(validation_results):
            successful_tests = sum(validation_results)
            total_tests = len(validation_results)
            if successful_tests == total_tests:
                logger.info("NVIDIA NVENC hardware acceleration validated successfully")
                # Mark NVIDIA NVENC encoders as validated
                if encoders.get('h264_nvenc'):
                    self.validation_results['h264_nvenc'] = True
                if encoders.get('hevc_nvenc'):
                    self.validation_results['hevc_nvenc'] = True
            else:
                logger.warning(f"NVIDIA NVENC validation partially successful ({successful_tests}/{total_tests} tests passed)")
                logger.info("NVIDIA NVENC encoder will be used but may fallback to software encoding on errors")
                # Mark as partially validated
                if encoders.get('h264_nvenc'):
                    self.validation_results['h264_nvenc'] = 'partial'
                if encoders.get('hevc_nvenc'):
                    self.validation_results['hevc_nvenc'] = 'partial'
        else:
            logger.warning("All NVIDIA NVENC validation tests failed - encoder detected but may not function properly")
            logger.info("Software fallback will be used automatically if NVIDIA NVENC fails during encoding")
            # Disable NVIDIA NVENC encoders that failed validation
            if encoders.get('h264_nvenc'):
                encoders['h264_nvenc'] = False
                self.validation_results['h264_nvenc'] = False
                logger.info("Disabled h264_nvenc encoder due to validation failure")
            if encoders.get('hevc_nvenc'):
                encoders['hevc_nvenc'] = False
                self.validation_results['hevc_nvenc'] = False
                logger.info("Disabled hevc_nvenc encoder due to validation failure")
    
    def is_encoder_validated(self, encoder: str) -> bool:
        """Check if an encoder has been validated and is working properly"""
        if not hasattr(self, 'validation_results'):
            return True  # Assume valid if not tested
        if encoder not in self.validation_results:
            return True  # Assume valid if not tested
        return self.validation_results[encoder] is True
    
    def get_best_encoder(self, codec: str = "h264") -> Tuple[str, str]:
        """
        Get the best available encoder for the specified codec
        Returns: (encoder_name, acceleration_type)
        """
        if codec.lower() == "h264":
            # Priority order: NVIDIA -> AMD -> Intel -> Software
            # Only use encoders that are available AND validated
            if (self.ffmpeg_encoders.get('h264_nvenc') and 
                self.has_nvidia_gpu() and 
                self.is_encoder_validated('h264_nvenc')):
                return 'h264_nvenc', 'nvidia'
            elif (self.ffmpeg_encoders.get('h264_amf') and 
                  self.has_amd_gpu() and 
                  self.is_encoder_validated('h264_amf')):
                return 'h264_amf', 'amd'
            elif self.ffmpeg_encoders.get('h264_qsv'):
                return 'h264_qsv', 'intel'
            elif self.ffmpeg_encoders.get('libx264'):
                return 'libx264', 'software'
        
        elif codec.lower() == "h265" or codec.lower() == "hevc":
            if (self.ffmpeg_encoders.get('hevc_nvenc') and 
                self.has_nvidia_gpu() and 
                self.is_encoder_validated('hevc_nvenc')):
                return 'hevc_nvenc', 'nvidia'
            elif (self.ffmpeg_encoders.get('hevc_amf') and 
                  self.has_amd_gpu() and 
                  self.is_encoder_validated('hevc_amf')):
                return 'hevc_amf', 'amd'
            elif self.ffmpeg_encoders.get('hevc_qsv'):
                return 'hevc_qsv', 'intel'
            elif self.ffmpeg_encoders.get('libx265'):
                return 'libx265', 'software'
        
        elif codec.lower() == "av1":
            if (self.ffmpeg_encoders.get('av1_amf') and 
                self.has_amd_gpu() and 
                self.is_encoder_validated('av1_amf')):
                return 'av1_amf', 'amd'
            # Add other AV1 encoders in the future
        
        # Fallback to software encoding
        logger.warning(f"No hardware acceleration available for {codec}, falling back to software")
        return 'libx264' if codec.lower() == "h264" else 'libx265', 'software'
    
    def get_amd_encoder_options(self, encoder: str) -> List[str]:
        """Get AMD AMF-specific encoder options for better performance and compatibility"""
        options = []
        
        if encoder == 'h264_amf':
            # Ultra-conservative AMD AMF options for maximum compatibility
            # Avoid advanced features that might cause "Invalid argument" errors
            options.extend([
                '-usage', 'transcoding',      # Standard transcoding usage
                '-quality', 'speed',          # Speed quality mode (most compatible)
                '-rc', 'cbr',                 # Constant bitrate (most stable)
                '-enforce_hrd', '1',          # Enforce HRD compliance
                '-filler_data', '1',          # Add filler data for bitrate compliance
                '-frame_skipping', '0',       # Disable frame skipping
                '-vbaq', '0',                 # Disable VBAQ (can cause issues)
                '-preanalysis', '0'           # Disable preanalysis (can cause issues)
            ])
        elif encoder == 'hevc_amf':
            # Conservative HEVC AMF options
            options.extend([
                '-usage', 'transcoding',
                '-quality', 'speed',
                '-rc', 'cbr',
                '-enforce_hrd', '1',
                '-filler_data', '1',
                '-frame_skipping', '0'
            ])
        elif encoder == 'av1_amf':
            # Conservative AV1 AMF options (minimal settings)
            options.extend([
                '-usage', 'transcoding',
                '-quality', 'speed'
            ])
        
        return options
    
    def has_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available"""
        return any(gpu['vendor'] == GPUVendor.NVIDIA.value for gpu in self.gpu_info)
    
    def has_amd_gpu(self) -> bool:
        """Check if AMD GPU is available"""
        return any(gpu['vendor'] == GPUVendor.AMD.value for gpu in self.gpu_info)
    
    def has_hardware_acceleration(self) -> bool:
        """Check if any hardware acceleration is available"""
        hardware_encoders = ['h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf', 'av1_amf', 'h264_qsv', 'hevc_qsv']
        return any(self.ffmpeg_encoders.get(encoder, False) for encoder in hardware_encoders)
    
    def force_software_encoding(self) -> None:
        """Force software encoding by disabling all hardware encoders"""
        logger.info("Forcing software encoding mode")
        for encoder in ['h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf', 'av1_amf', 'h264_qsv', 'hevc_qsv']:
            if encoder in self.ffmpeg_encoders:
                self.ffmpeg_encoders[encoder] = False
                self.validation_results[encoder] = False
        logger.info("All hardware encoders disabled - will use software encoding only")
    
    def get_system_report(self) -> str:
        """Generate a comprehensive system report"""
        report = []
        report.append("=== Hardware Detection Report ===")
        report.append(f"Platform: {self.system_info['platform']} ({self.system_info['architecture']})")
        report.append(f"CPU Cores: {self.system_info['cpu_count']}")
        report.append(f"Memory: {self.system_info['memory_gb']} GB")
        report.append("")
        
        report.append("=== GPU Information ===")
        if self.gpu_info:
            for gpu in self.gpu_info:
                memory_info = f" ({gpu['memory_mb']} MB)" if gpu['memory_mb'] != 'Unknown' else ""
                report.append(f"  {gpu['vendor'].upper()}: {gpu['name']}{memory_info}")
        else:
            report.append("  No GPUs detected")
        report.append("")
        
        report.append("=== Available Hardware Encoders ===")
        hardware_encoders = ['h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf', 'av1_amf', 'h264_qsv', 'hevc_qsv']
        available_hw_encoders = [k for k in hardware_encoders if self.ffmpeg_encoders.get(k, False)]
        
        if available_hw_encoders:
            for encoder in available_hw_encoders:
                vendor_map = {
                    'nvenc': 'NVIDIA',
                    'amf': 'AMD',
                    'qsv': 'Intel'
                }
                vendor = next((v for k, v in vendor_map.items() if k in encoder), 'Unknown')
                if not hasattr(self, 'validation_results'):
                    status = "? Not tested"
                else:
                    validation_status = self.validation_results.get(encoder, 'Not tested')
                    if validation_status is True:
                        status = "OK Validated"
                    elif validation_status is False:
                        status = "✗ Failed validation"
                    elif validation_status == 'partial':
                        status = "⚠ Partially validated"
                    else:
                        status = "? Not tested"
                report.append(f"  {status} {encoder} ({vendor})")
        else:
            report.append("  No hardware encoders detected")
        
        report.append("")
        report.append("=== Available Software Encoders ===")
        software_encoders = ['libx264', 'libx265']
        available_sw_encoders = [k for k in software_encoders if self.ffmpeg_encoders.get(k, False)]
        
        if available_sw_encoders:
            for encoder in available_sw_encoders:
                report.append(f"  OK {encoder}")
        else:
            report.append("  No software encoders detected")
        
        unavailable_encoders = [k for k, v in self.ffmpeg_encoders.items() if not v]
        if unavailable_encoders:
            report.append("")
            report.append("=== Unavailable Encoders ===")
            for encoder in unavailable_encoders:
                report.append(f"  ✗ {encoder}")
        
        report.append("")
        best_h264, accel_type = self.get_best_encoder("h264")
        best_h265, accel_type_h265 = self.get_best_encoder("h265")
        report.append(f"Best H.264 Encoder: {best_h264} ({accel_type})")
        report.append(f"Best H.265 Encoder: {best_h265} ({accel_type_h265})")
        
        # Add AV1 if available
        if self.ffmpeg_encoders.get('av1_amf'):
            best_av1, accel_type_av1 = self.get_best_encoder("av1")
            report.append(f"Best AV1 Encoder: {best_av1} ({accel_type_av1})")
        
        # Add hardware acceleration status
        report.append("")
        if self.has_hardware_acceleration():
            report.append("🚀 Hardware Acceleration: ENABLED")
        else:
            report.append("⚠️  Hardware Acceleration: NOT AVAILABLE")
        
        return "\n".join(report) 