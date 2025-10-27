"""
Quality Gates Module
Implements VMAF and SSIM quality measurements for CAE pipeline
"""

import os
import subprocess
import json
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class QualityGates:
    """Objective quality measurement using VMAF and SSIM."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self._vmaf_available = None
    
    def check_vmaf_available(self) -> bool:
        """Check if libvmaf is available in FFmpeg (cached)."""
        if self._vmaf_available is None:
            try:
                from .ffmpeg_utils import FFmpegUtils
                self._vmaf_available = FFmpegUtils.check_libvmaf_available()
                if self._vmaf_available:
                    logger.info("libvmaf filter detected in FFmpeg")
                else:
                    logger.warning("libvmaf not available; will use SSIM fallback")
            except Exception as e:
                logger.warning(f"Failed to check libvmaf availability: {e}")
                self._vmaf_available = False
        return self._vmaf_available
    
    def evaluate_quality(self, original_path: str, compressed_path: str,
                        vmaf_threshold: float = 80.0, ssim_threshold: float = 0.94,
                        eval_height: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate compressed video quality against original.
        
        Args:
            original_path: Reference video
            compressed_path: Compressed video to evaluate
            vmaf_threshold: Minimum VMAF score (0-100)
            ssim_threshold: Minimum SSIM score (0-1)
            eval_height: Downscale eval to this height (e.g., 540); None = native
        
        Returns:
            Dict with keys:
                - vmaf_score: float or None
                - ssim_score: float or None
                - passes: bool (meets thresholds)
                - method: 'vmaf+ssim', 'ssim_only', or 'error'
                - details: additional info
        """
        result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': False,
            'method': 'error',
            'details': {}
        }
        
        try:
            # Validate inputs
            if not os.path.exists(original_path):
                result['details']['error'] = f"Original not found: {original_path}"
                return result
            if not os.path.exists(compressed_path):
                result['details']['error'] = f"Compressed not found: {compressed_path}"
                return result
            
            # Check VMAF availability
            has_vmaf = self.check_vmaf_available()
            
            if has_vmaf:
                # Try VMAF + SSIM
                vmaf_score = self._compute_vmaf(original_path, compressed_path, eval_height)
                ssim_score = self._compute_ssim(original_path, compressed_path, eval_height)
                
                result['vmaf_score'] = vmaf_score
                result['ssim_score'] = ssim_score
                result['method'] = 'vmaf+ssim'
                
                # Both must pass
                vmaf_pass = (vmaf_score is not None and vmaf_score >= vmaf_threshold)
                ssim_pass = (ssim_score is not None and ssim_score >= ssim_threshold)
                result['passes'] = vmaf_pass and ssim_pass
                
                result['details']['vmaf_threshold'] = vmaf_threshold
                result['details']['ssim_threshold'] = ssim_threshold
                result['details']['vmaf_pass'] = vmaf_pass
                result['details']['ssim_pass'] = ssim_pass
                
            else:
                # SSIM-only fallback
                ssim_score = self._compute_ssim(original_path, compressed_path, eval_height)
                result['ssim_score'] = ssim_score
                result['method'] = 'ssim_only'
                result['passes'] = (ssim_score is not None and ssim_score >= ssim_threshold)
                result['details']['ssim_threshold'] = ssim_threshold
            
            if eval_height:
                result['details']['eval_height'] = eval_height
            
            logger.info(f"Quality eval: method={result['method']}, "
                       f"VMAF={result['vmaf_score']:.2f if result['vmaf_score'] else 'N/A'}, "
                       f"SSIM={result['ssim_score']:.4f if result['ssim_score'] else 'N/A'}, "
                       f"passes={result['passes']}")
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            result['details']['error'] = str(e)
        
        return result
    
    def _compute_vmaf(self, ref_path: str, dist_path: str, 
                     eval_height: Optional[int] = None) -> Optional[float]:
        """Compute VMAF score using FFmpeg libvmaf filter.
        
        Returns VMAF score (0-100) or None on error.
        """
        try:
            # Build filter chain: scale both to eval resolution, then libvmaf
            scale_filter = ""
            if eval_height:
                scale_filter = f"scale=-2:{eval_height}:flags=bicubic,"
            
            # libvmaf filter: [distorted][reference]libvmaf
            filter_complex = (
                f"[0:v]{scale_filter}setpts=PTS-STARTPTS[dist];"
                f"[1:v]{scale_filter}setpts=PTS-STARTPTS[ref];"
                f"[dist][ref]libvmaf=log_fmt=json:log_path=-:n_threads=4"
            )
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', dist_path,
                '-i', ref_path,
                '-filter_complex', filter_complex,
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                logger.warning(f"VMAF computation failed: {result.stderr}")
                return None
            
            # Parse JSON from stderr (libvmaf outputs to stderr)
            vmaf_data = None
            for line in result.stderr.split('\n'):
                if 'VMAF score' in line or '"vmaf"' in line.lower():
                    # Try to extract JSON
                    try:
                        # Find JSON object in line
                        start = line.find('{')
                        if start >= 0:
                            vmaf_data = json.loads(line[start:])
                            break
                    except:
                        continue
            
            if vmaf_data and 'pooled_metrics' in vmaf_data:
                vmaf_score = vmaf_data['pooled_metrics']['vmaf']['mean']
                return float(vmaf_score)
            
            # Fallback: look for simple numeric output
            for line in result.stderr.split('\n'):
                if 'vmaf' in line.lower() and 'mean' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'mean' in part.lower() and i + 1 < len(parts):
                            try:
                                return float(parts[i + 1].strip(':,'))
                            except:
                                pass
            
            logger.warning("Could not parse VMAF score from output")
            return None
            
        except subprocess.TimeoutExpired:
            logger.warning("VMAF computation timed out")
            return None
        except Exception as e:
            logger.warning(f"VMAF computation error: {e}")
            return None
    
    def _compute_ssim(self, ref_path: str, dist_path: str,
                     eval_height: Optional[int] = None) -> Optional[float]:
        """Compute SSIM score using FFmpeg ssim filter (Y channel).
        
        Returns SSIM score (0-1) or None on error.
        """
        try:
            # Build filter chain
            scale_filter = ""
            if eval_height:
                scale_filter = f"scale=-2:{eval_height}:flags=bicubic,"
            
            filter_complex = (
                f"[0:v]{scale_filter}setpts=PTS-STARTPTS[dist];"
                f"[1:v]{scale_filter}setpts=PTS-STARTPTS[ref];"
                f"[dist][ref]ssim=stats_file=-"
            )
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', dist_path,
                '-i', ref_path,
                '-filter_complex', filter_complex,
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode != 0:
                logger.warning(f"SSIM computation failed: {result.stderr}")
                return None
            
            # Parse SSIM from stderr (ssim stats go to stderr)
            # Format: "n:1 Y:0.95123 U:... V:... All:0.94567 (12.34dB)"
            ssim_scores = []
            for line in result.stderr.split('\n'):
                if line.startswith('n:') and 'Y:' in line:
                    try:
                        y_part = line.split('Y:')[1].split()[0]
                        ssim_scores.append(float(y_part))
                    except:
                        continue
            
            if ssim_scores:
                # Return average Y-channel SSIM
                avg_ssim = sum(ssim_scores) / len(ssim_scores)
                return avg_ssim
            
            logger.warning("Could not parse SSIM score from output")
            return None
            
        except subprocess.TimeoutExpired:
            logger.warning("SSIM computation timed out")
            return None
        except Exception as e:
            logger.warning(f"SSIM computation error: {e}")
            return None

