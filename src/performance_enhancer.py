"""
Performance Enhancement Module
Implements caching, parallel processing, and performance optimizations
"""

import os
import json
import pickle
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable
import logging
import time
from functools import wraps, lru_cache
from pathlib import Path
import psutil
from .ffmpeg_utils import FFmpegUtils

logger = logging.getLogger(__name__)

class PerformanceEnhancer:
    def __init__(self, config_manager):
        self.config = config_manager
        self.cache_dir = os.path.join(self.config.get_temp_dir(), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Performance monitoring
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0,
            'parallel_jobs_completed': 0
        }
        
        # System optimization
        self.cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        self.logical_cpu_count = psutil.cpu_count(logical=True)
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Optimal worker counts
        self.optimal_process_workers = min(self.cpu_count, 4)  # Don't overwhelm system
        self.optimal_thread_workers = min(self.logical_cpu_count, 8)
        
        logger.info(f"Performance enhancer initialized: {self.cpu_count} cores, "
                   f"{self.memory_gb:.1f}GB RAM, {self.optimal_process_workers} process workers")
    
    def cached_operation(self, cache_key_func: Callable = None, ttl_hours: int = 24):
        """
        Decorator for caching expensive operations
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Check cache
                cached_result = self._get_from_cache(cache_key, ttl_hours)
                if cached_result is not None:
                    self.performance_stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function
                self.performance_stats['cache_misses'] += 1
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self.performance_stats['total_processing_time'] += execution_time
                
                # Cache result
                self._save_to_cache(cache_key, result)
                logger.debug(f"Cached result for {func.__name__} (took {execution_time:.2f}s)")
                
                return result
            
            return wrapper
        return decorator
    
    def parallel_process_videos(self, video_tasks: List[Dict[str, Any]], 
                              processing_func: Callable) -> List[Dict[str, Any]]:
        """
        Process multiple videos in parallel using process pool
        """
        
        logger.info(f"Starting parallel processing of {len(video_tasks)} videos")
        
        # Optimize worker count based on task complexity and system resources
        worker_count = self._calculate_optimal_workers(video_tasks, 'process')
        
        results = []
        failed_tasks = []
        
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(processing_func, task): task 
                for task in video_tasks
            }
            
            # Collect results with progress tracking
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed += 1
                
                try:
                    result = future.result(timeout=600)  # 10 minute timeout per video
                    results.append(result)
                    self.performance_stats['parallel_jobs_completed'] += 1
                    
                    logger.info(f"Completed {completed}/{len(video_tasks)}: {task.get('input_path', 'unknown')}")
                    
                except Exception as e:
                    logger.error(f"Task failed for {task.get('input_path', 'unknown')}: {e}")
                    failed_tasks.append({'task': task, 'error': str(e)})
        
        logger.info(f"Parallel processing completed: {len(results)} successful, {len(failed_tasks)} failed")
        
        return {
            'successful': results,
            'failed': failed_tasks,
            'stats': {
                'total_tasks': len(video_tasks),
                'successful_count': len(results),
                'failed_count': len(failed_tasks),
                'worker_count': worker_count
            }
        }
    
    def parallel_process_lightweight(self, tasks: List[Dict[str, Any]], 
                                   processing_func: Callable) -> List[Dict[str, Any]]:
        """
        Process lightweight tasks in parallel using thread pool
        """
        
        worker_count = self._calculate_optimal_workers(tasks, 'thread')
        
        results = []
        
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {
                executor.submit(processing_func, task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result(timeout=120)  # 2 minute timeout for lightweight tasks
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Lightweight task failed: {e}")
        
        return results
    
    def optimize_system_resources(self):
        """
        Optimize system resources for video processing
        """
        
        logger.info("Optimizing system resources...")
        
        optimizations = {
            'process_priority': 'normal',
            'io_priority': 'normal',
            'memory_optimization': False,
            'cpu_affinity': None
        }
        
        try:
            current_process = psutil.Process()
            
            # Set process priority for video processing
            if self.memory_gb >= 8:  # Sufficient RAM
                try:
                    current_process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 5)
                    optimizations['process_priority'] = 'below_normal'
                except:
                    pass
            
            # Set I/O priority for better disk performance
            try:
                if hasattr(current_process, 'ionice'):
                    current_process.ionice(psutil.IOPRIO_CLASS_BE, value=4)
                    optimizations['io_priority'] = 'best_effort'
            except:
                pass
            
            # Memory optimization for large files
            if self.memory_gb < 4:
                optimizations['memory_optimization'] = True
                logger.warning("Low memory detected - enabling memory optimization mode")
            
            # CPU affinity optimization
            if self.cpu_count >= 4:
                try:
                    # Reserve one core for system processes
                    available_cores = list(range(1, self.cpu_count))
                    current_process.cpu_affinity(available_cores)
                    optimizations['cpu_affinity'] = available_cores
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"System optimization failed: {e}")
        
        return optimizations
    
    def intelligent_batch_processing(self, file_list: List[str], 
                                   processing_func: Callable,
                                   batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Intelligently batch process files based on system capabilities
        """
        
        if not batch_size:
            batch_size = self._calculate_optimal_batch_size(file_list)
        
        logger.info(f"Processing {len(file_list)} files in batches of {batch_size}")
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(file_list), batch_size):
            batch = file_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(file_list) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            # Create tasks for this batch
            batch_tasks = [{'input_path': file_path, 'batch_id': batch_num} for file_path in batch]
            
            # Process batch
            batch_results = self.parallel_process_videos(batch_tasks, processing_func)
            all_results.extend(batch_results['successful'])
            
            # Memory cleanup between batches
            if batch_num < total_batches:
                self._cleanup_between_batches()
        
        return all_results
    
    def adaptive_quality_processing(self, input_path: str, target_size_mb: float,
                                  quality_levels: List[str] = None) -> Dict[str, Any]:
        """
        Adaptively process video with different quality levels and select best result
        """
        
        if not quality_levels:
            quality_levels = ['fast', 'balanced', 'high_quality']
        
        logger.info(f"Adaptive processing with quality levels: {quality_levels}")
        
        # Create processing tasks for different quality levels
        tasks = []
        for quality in quality_levels:
            task = {
                'input_path': input_path,
                'target_size_mb': target_size_mb,
                'quality_level': quality,
                'temp_output': os.path.join(self.config.get_temp_dir(), f"adaptive_{quality}_{int(time.time())}.mp4")
            }
            tasks.append(task)
        
        # Process in parallel
        results = self.parallel_process_lightweight(tasks, self._process_quality_level)
        
        # Select best result
        best_result = self._select_best_quality_result(results, target_size_mb)
        
        # Cleanup other results
        for result in results:
            if result != best_result and os.path.exists(result.get('output_path', '')):
                os.remove(result['output_path'])
        
        return best_result
    
    def memory_efficient_processing(self, input_path: str, processing_func: Callable,
                                  chunk_size_mb: int = 100) -> Dict[str, Any]:
        """
        Process large files in memory-efficient chunks
        """
        
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        
        if file_size_mb <= chunk_size_mb or self.memory_gb >= 8:
            # Process normally if file is small or we have enough memory
            return processing_func(input_path)
        
        logger.info(f"Memory-efficient processing for {file_size_mb:.1f}MB file")
        
        # Split processing into segments
        # This is a conceptual implementation - actual implementation would depend on the processing type
        segments = max(2, int(file_size_mb / chunk_size_mb))
        segment_duration = self._get_video_duration(input_path) / segments
        
        segment_results = []
        temp_segments = []
        
        try:
            # Process each segment
            for i in range(segments):
                start_time = i * segment_duration
                segment_output = os.path.join(self.config.get_temp_dir(), f"segment_{i}.mp4")
                
                # Extract segment
                self._extract_video_segment(input_path, segment_output, start_time, segment_duration)
                temp_segments.append(segment_output)
                
                # Process segment
                segment_result = processing_func(segment_output)
                segment_results.append(segment_result)
                
                # Cleanup segment immediately to save memory
                if os.path.exists(segment_output):
                    os.remove(segment_output)
            
            # Combine results
            final_result = self._combine_segment_results(segment_results)
            
        finally:
            # Cleanup any remaining temporary files
            for temp_file in temp_segments:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return final_result
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        
        # Create a string representation of arguments
        key_data = {
            'function': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str, ttl_hours: int) -> Optional[Any]:
        """Retrieve item from cache if it exists and is not expired"""
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.cache")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Check if cache is expired
            file_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            if file_age_hours > ttl_hours:
                os.remove(cache_file)
                return None
            
            # Load cached data
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_key}: {e}")
            # Remove corrupted cache file
            try:
                os.remove(cache_file)
            except:
                pass
            return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache"""
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.cache")
        
        try:
            # Ensure cache directory exists even if it was removed mid-run
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache save failed for {cache_key}: {e}")
    
    def _calculate_optimal_workers(self, tasks: List[Dict[str, Any]], worker_type: str) -> int:
        """Calculate optimal number of workers based on tasks and system resources"""
        
        task_count = len(tasks)
        
        if worker_type == 'process':
            # For CPU-intensive video processing
            base_workers = self.optimal_process_workers
            
            # Adjust based on available memory
            memory_factor = min(self.memory_gb / 4.0, 2.0)  # Scale with memory up to 8GB
            memory_adjusted = int(base_workers * memory_factor)
            
            # Don't exceed task count
            return min(memory_adjusted, task_count, 6)  # Max 6 processes to avoid overwhelming
        
        else:  # thread
            # For I/O-bound or lightweight processing
            base_workers = self.optimal_thread_workers
            
            # Threads can be more numerous
            return min(base_workers, task_count, 12)  # Max 12 threads
    
    def _calculate_optimal_batch_size(self, file_list: List[str]) -> int:
        """Calculate optimal batch size based on system resources and file sizes"""
        
        # Estimate average file size
        sample_size = min(5, len(file_list))
        total_sample_size = 0
        
        for file_path in file_list[:sample_size]:
            if os.path.exists(file_path):
                total_sample_size += os.path.getsize(file_path)
        
        if sample_size > 0:
            avg_file_size_mb = (total_sample_size / sample_size) / (1024 * 1024)
        else:
            avg_file_size_mb = 100  # Default assumption
        
        # Calculate batch size based on memory
        memory_per_batch = self.memory_gb * 0.6  # Use 60% of available memory
        files_per_gb = max(1, int(1024 / avg_file_size_mb))
        
        optimal_batch_size = max(2, int(memory_per_batch * files_per_gb))
        
        # Practical limits
        return min(optimal_batch_size, 20, len(file_list))
    
    def _cleanup_between_batches(self):
        """Cleanup resources between batch processing"""
        
        # Clear temporary files
        temp_dir = self.config.get_temp_dir()
        temp_files = [f for f in os.listdir(temp_dir) if f.startswith('temp_') or f.startswith('candidate_')]
        
        for temp_file in temp_files:
            try:
                file_path = os.path.join(temp_dir, temp_file)
                if os.path.isfile(file_path):
                    # Only remove files older than 10 minutes
                    if time.time() - os.path.getmtime(file_path) > 600:
                        os.remove(file_path)
            except:
                pass
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def _process_quality_level(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single quality level task"""
        
        # This is a placeholder - in real implementation, you'd call the actual processing function
        # with the specified quality level
        
        quality_settings = {
            'fast': {'crf': 28, 'preset': 'veryfast'},
            'balanced': {'crf': 23, 'preset': 'medium'},
            'high_quality': {'crf': 18, 'preset': 'slow'}
        }
        
        settings = quality_settings.get(task['quality_level'], quality_settings['balanced'])
        
        # Simulate processing
        time.sleep(0.1)  # Placeholder processing time
        
        return {
            'quality_level': task['quality_level'],
            'output_path': task['temp_output'],
            'settings': settings,
            'estimated_quality_score': {'fast': 6, 'balanced': 8, 'high_quality': 9}[task['quality_level']],
            'processing_time': 0.1
        }
    
    def _select_best_quality_result(self, results: List[Dict[str, Any]], target_size_mb: float) -> Dict[str, Any]:
        """Select the best quality result from adaptive processing"""
        
        if not results:
            raise RuntimeError("No quality processing results available")
        
        # Score each result
        scored_results = []
        
        for result in results:
            score = result.get('estimated_quality_score', 5)
            
            # Bonus for faster processing
            processing_time = result.get('processing_time', 1.0)
            speed_bonus = max(0, (2.0 - processing_time) * 0.5)
            
            total_score = score + speed_bonus
            scored_results.append((total_score, result))
        
        # Return highest scoring result
        best_score, best_result = max(scored_results, key=lambda x: x[0])
        
        logger.info(f"Selected quality level '{best_result['quality_level']}' with score {best_score:.1f}")
        
        return best_result
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration using shared utility"""
        return FFmpegUtils.get_video_duration(video_path)
    
    def _extract_video_segment(self, input_path: str, output_path: str, 
                             start_time: float, duration: float):
        """Extract a segment from video using shared utility"""
        return FFmpegUtils.extract_video_segment(input_path, output_path, start_time, duration)
    
    def _combine_segment_results(self, segment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple segments"""
        
        # This is a placeholder - actual implementation would depend on the type of results
        combined_result = {
            'segments_processed': len(segment_results),
            'total_processing_time': sum(r.get('processing_time', 0) for r in segment_results),
            'combined_output': 'segments_combined.mp4'  # Placeholder
        }
        
        return combined_result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        
        cache_hit_rate = 0
        if self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'] > 0:
            cache_hit_rate = self.performance_stats['cache_hits'] / (
                self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']
            ) * 100
        
        return {
            'cache_hit_rate_percent': cache_hit_rate,
            'total_cache_hits': self.performance_stats['cache_hits'],
            'total_cache_misses': self.performance_stats['cache_misses'],
            'total_processing_time_seconds': self.performance_stats['total_processing_time'],
            'parallel_jobs_completed': self.performance_stats['parallel_jobs_completed'],
            'system_info': {
                'cpu_cores': self.cpu_count,
                'logical_cores': self.logical_cpu_count,
                'memory_gb': self.memory_gb,
                'optimal_process_workers': self.optimal_process_workers,
                'optimal_thread_workers': self.optimal_thread_workers
            }
        }
    
    def clear_cache(self, older_than_hours: int = 0):
        """Clear cache files"""
        
        cleared_count = 0
        current_time = time.time()
        
        for cache_file in os.listdir(self.cache_dir):
            if cache_file.endswith('.cache'):
                file_path = os.path.join(self.cache_dir, cache_file)
                
                try:
                    file_age_hours = (current_time - os.path.getmtime(file_path)) / 3600
                    
                    if file_age_hours > older_than_hours:
                        os.remove(file_path)
                        cleared_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {cleared_count} cache files older than {older_than_hours} hours")
        return cleared_count 