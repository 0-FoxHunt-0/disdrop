"""
Performance caching system for video compression results.

This module provides efficient caching and retrieval of expensive computation results
to avoid redundant processing and improve overall performance.
"""

import json
import os
import pickle
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import threading
try:
    from .video_fingerprinter import VideoFingerprinter, CompressionParams, CacheKeyGenerator
    from .logger_setup import get_logger
except ImportError:
    # Fallback for direct execution
    from video_fingerprinter import VideoFingerprinter, CompressionParams, CacheKeyGenerator
    from logger_setup import get_logger

logger = get_logger(__name__)


@dataclass
class QualityResult:
    """Represents quality evaluation results."""
    vmaf_score: Optional[float]
    ssim_score: Optional[float]
    psnr_score: Optional[float]
    computation_time: float
    evaluation_method: str  # 'full', 'fast', 'predicted'
    confidence: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityResult':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CompressionResult:
    """Represents compression operation results."""
    output_file_size: int
    compression_ratio: float
    processing_time: float
    quality_result: Optional[QualityResult]
    success: bool
    error_message: Optional[str]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.quality_result:
            data['quality_result'] = self.quality_result.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionResult':
        """Create from dictionary."""
        quality_data = data.pop('quality_result', None)
        quality_result = QualityResult.from_dict(quality_data) if quality_data else None
        return cls(quality_result=quality_result, **data)


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    cache_key: str
    video_fingerprint: str
    compression_params: Dict[str, Any]
    result: CompressionResult
    access_count: int
    last_accessed: float
    created_at: float
    expires_at: Optional[float]
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cache_key': self.cache_key,
            'video_fingerprint': self.video_fingerprint,
            'compression_params': self.compression_params,
            'result': self.result.to_dict(),
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'created_at': self.created_at,
            'expires_at': self.expires_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        result_data = data.pop('result')
        result = CompressionResult.from_dict(result_data)
        return cls(result=result, **data)


class PerformanceCache:
    """High-performance caching system for video compression results."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 500):
        """
        Initialize the performance cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.disdrop' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.db_path = self.cache_dir / 'performance_cache.db'
        
        self.fingerprinter = VideoFingerprinter()
        self.key_generator = CacheKeyGenerator(self.fingerprinter)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_lookups = 0
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Performance cache initialized at {self.cache_dir}")
    
    def _init_database(self):
        """Initialize SQLite database for cache storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        cache_key TEXT PRIMARY KEY,
                        video_fingerprint TEXT NOT NULL,
                        compression_params TEXT NOT NULL,
                        result_data BLOB NOT NULL,
                        access_count INTEGER DEFAULT 1,
                        last_accessed REAL NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL,
                        data_size INTEGER NOT NULL
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_fingerprint 
                    ON cache_entries(video_fingerprint)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_last_accessed 
                    ON cache_entries(last_accessed)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_expires_at 
                    ON cache_entries(expires_at)
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error initializing cache database: {e}")
    
    def get_cached_quality_result(self, video_path: str, params: CompressionParams) -> Optional[QualityResult]:
        """
        Retrieve cached quality result for video and parameters.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            
        Returns:
            Cached quality result if found, None otherwise
        """
        with self._lock:
            self._total_lookups += 1
            
            try:
                cache_key = self.key_generator.generate_cache_key(video_path, params)
                entry = self._get_cache_entry(cache_key)
                
                if entry and not entry.is_expired():
                    # Update access statistics
                    self._update_access_stats(cache_key)
                    self._cache_hits += 1
                    
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return entry.result.quality_result
                
                self._cache_misses += 1
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
                
            except Exception as e:
                logger.error(f"Error retrieving cached quality result: {e}")
                self._cache_misses += 1
                return None
    
    def cache_quality_result(self, video_path: str, params: CompressionParams, 
                           result: QualityResult, ttl_hours: int = 24) -> None:
        """
        Cache quality result for video and parameters.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            result: Quality result to cache
            ttl_hours: Time to live in hours
        """
        with self._lock:
            try:
                cache_key = self.key_generator.generate_cache_key(video_path, params)
                fingerprint = self.fingerprinter.generate_fingerprint(video_path)
                
                # Create compression result wrapper
                compression_result = CompressionResult(
                    output_file_size=0,  # Not applicable for quality-only results
                    compression_ratio=0.0,
                    processing_time=result.computation_time,
                    quality_result=result,
                    success=True,
                    error_message=None,
                    timestamp=time.time()
                )
                
                # Create cache entry
                expires_at = time.time() + (ttl_hours * 3600) if ttl_hours > 0 else None
                entry = CacheEntry(
                    cache_key=cache_key,
                    video_fingerprint=fingerprint.content_hash,
                    compression_params=params.__dict__,
                    result=compression_result,
                    access_count=1,
                    last_accessed=time.time(),
                    created_at=time.time(),
                    expires_at=expires_at
                )
                
                self._store_cache_entry(entry)
                logger.debug(f"Cached quality result for key: {cache_key}")
                
            except Exception as e:
                logger.error(f"Error caching quality result: {e}")
    
    def get_cached_compression_result(self, video_path: str, params: CompressionParams) -> Optional[CompressionResult]:
        """
        Retrieve cached compression result for video and parameters.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            
        Returns:
            Cached compression result if found, None otherwise
        """
        with self._lock:
            self._total_lookups += 1
            
            try:
                cache_key = self.key_generator.generate_cache_key(video_path, params)
                entry = self._get_cache_entry(cache_key)
                
                if entry and not entry.is_expired():
                    self._update_access_stats(cache_key)
                    self._cache_hits += 1
                    
                    logger.debug(f"Cache hit for compression result: {cache_key}")
                    return entry.result
                
                self._cache_misses += 1
                return None
                
            except Exception as e:
                logger.error(f"Error retrieving cached compression result: {e}")
                self._cache_misses += 1
                return None
    
    def cache_compression_result(self, video_path: str, params: CompressionParams, 
                               result: CompressionResult, ttl_hours: int = 24) -> None:
        """
        Cache compression result for video and parameters.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            result: Compression result to cache
            ttl_hours: Time to live in hours
        """
        with self._lock:
            try:
                cache_key = self.key_generator.generate_cache_key(video_path, params)
                fingerprint = self.fingerprinter.generate_fingerprint(video_path)
                
                expires_at = time.time() + (ttl_hours * 3600) if ttl_hours > 0 else None
                entry = CacheEntry(
                    cache_key=cache_key,
                    video_fingerprint=fingerprint.content_hash,
                    compression_params=params.__dict__,
                    result=result,
                    access_count=1,
                    last_accessed=time.time(),
                    created_at=time.time(),
                    expires_at=expires_at
                )
                
                self._store_cache_entry(entry)
                logger.debug(f"Cached compression result for key: {cache_key}")
                
            except Exception as e:
                logger.error(f"Error caching compression result: {e}")
    
    def get_similar_video_results(self, video_path: str, similarity_threshold: float = 0.8) -> List[Tuple[float, CompressionResult]]:
        """
        Get cached results from similar videos based on perceptual hashing.
        
        Args:
            video_path: Path to the video file
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of (similarity_score, result) tuples sorted by similarity
        """
        with self._lock:
            try:
                target_fingerprint = self.fingerprinter.generate_fingerprint(video_path)
                target_hash = target_fingerprint.perceptual_hash
                
                similar_results = []
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT video_fingerprint, result_data 
                        FROM cache_entries 
                        WHERE expires_at IS NULL OR expires_at > ?
                    ''', (time.time(),))
                    
                    for fingerprint, result_data in cursor.fetchall():
                        # Calculate similarity using perceptual hash
                        similarity = self.fingerprinter.calculate_similarity(target_hash, fingerprint)
                        
                        if similarity >= similarity_threshold:
                            result = pickle.loads(result_data)
                            compression_result = CompressionResult.from_dict(result)
                            similar_results.append((similarity, compression_result))
                
                # Sort by similarity (highest first)
                similar_results.sort(key=lambda x: x[0], reverse=True)
                
                logger.debug(f"Found {len(similar_results)} similar video results")
                return similar_results
                
            except Exception as e:
                logger.error(f"Error finding similar video results: {e}")
                return []
    
    def predict_quality_from_cache(self, video_path: str, params: CompressionParams) -> Optional[QualityResult]:
        """
        Predict quality result based on similar cached videos.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            
        Returns:
            Predicted quality result if sufficient similar data exists
        """
        try:
            similar_results = self.get_similar_video_results(video_path, similarity_threshold=0.7)
            
            if len(similar_results) < 2:
                return None
            
            # Extract quality scores from similar results
            vmaf_scores = []
            ssim_scores = []
            psnr_scores = []
            weights = []
            
            for similarity, result in similar_results[:5]:  # Use top 5 similar results
                if result.quality_result:
                    quality = result.quality_result
                    if quality.vmaf_score is not None:
                        vmaf_scores.append(quality.vmaf_score)
                        weights.append(similarity)
                    if quality.ssim_score is not None:
                        ssim_scores.append(quality.ssim_score)
                    if quality.psnr_score is not None:
                        psnr_scores.append(quality.psnr_score)
            
            if not vmaf_scores and not ssim_scores:
                return None
            
            # Calculate weighted averages
            def weighted_average(values, weights):
                if not values:
                    return None
                total_weight = sum(weights[:len(values)])
                if total_weight == 0:
                    return sum(values) / len(values)
                return sum(v * w for v, w in zip(values, weights[:len(values)])) / total_weight
            
            predicted_vmaf = weighted_average(vmaf_scores, weights)
            predicted_ssim = weighted_average(ssim_scores, weights)
            predicted_psnr = weighted_average(psnr_scores, weights)
            
            # Calculate confidence based on number of similar results and similarity scores
            confidence = min(0.9, len(similar_results) * 0.15 + max(weights) * 0.5)
            
            predicted_result = QualityResult(
                vmaf_score=predicted_vmaf,
                ssim_score=predicted_ssim,
                psnr_score=predicted_psnr,
                computation_time=0.1,  # Very fast prediction
                evaluation_method='predicted',
                confidence=confidence,
                timestamp=time.time()
            )
            
            logger.debug(f"Predicted quality from {len(similar_results)} similar videos")
            return predicted_result
            
        except Exception as e:
            logger.error(f"Error predicting quality from cache: {e}")
            return None
    
    def _get_cache_entry(self, cache_key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT cache_key, video_fingerprint, compression_params, 
                           result_data, access_count, last_accessed, created_at, expires_at
                    FROM cache_entries 
                    WHERE cache_key = ?
                ''', (cache_key,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Deserialize data
                result_data = pickle.loads(row[3])
                compression_result = CompressionResult.from_dict(result_data)
                compression_params = json.loads(row[2])
                
                return CacheEntry(
                    cache_key=row[0],
                    video_fingerprint=row[1],
                    compression_params=compression_params,
                    result=compression_result,
                    access_count=row[4],
                    last_accessed=row[5],
                    created_at=row[6],
                    expires_at=row[7]
                )
                
        except Exception as e:
            logger.error(f"Error retrieving cache entry {cache_key}: {e}")
            return None
    
    def _store_cache_entry(self, entry: CacheEntry) -> None:
        """Store cache entry in database."""
        try:
            # Serialize result data
            result_data = pickle.dumps(entry.result.to_dict())
            compression_params = json.dumps(entry.compression_params)
            data_size = len(result_data) + len(compression_params)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, video_fingerprint, compression_params, result_data,
                     access_count, last_accessed, created_at, expires_at, data_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.cache_key,
                    entry.video_fingerprint,
                    compression_params,
                    result_data,
                    entry.access_count,
                    entry.last_accessed,
                    entry.created_at,
                    entry.expires_at,
                    data_size
                ))
                conn.commit()
            
            # Check if cache size limit is exceeded
            self._enforce_size_limit()
            
        except Exception as e:
            logger.error(f"Error storing cache entry {entry.cache_key}: {e}")
    
    def _update_access_stats(self, cache_key: str) -> None:
        """Update access statistics for cache entry."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE cache_entries 
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE cache_key = ?
                ''', (time.time(), cache_key))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating access stats for {cache_key}: {e}")
    
    def cleanup_expired_entries(self, max_age_hours: int = 24) -> int:
        """
        Remove expired cache entries.
        
        Args:
            max_age_hours: Maximum age for entries without explicit expiration
            
        Returns:
            Number of entries removed
        """
        with self._lock:
            try:
                current_time = time.time()
                age_threshold = current_time - (max_age_hours * 3600)
                
                with sqlite3.connect(self.db_path) as conn:
                    # Remove explicitly expired entries
                    cursor = conn.execute('''
                        DELETE FROM cache_entries 
                        WHERE expires_at IS NOT NULL AND expires_at < ?
                    ''', (current_time,))
                    expired_count = cursor.rowcount
                    
                    # Remove old entries without explicit expiration
                    cursor = conn.execute('''
                        DELETE FROM cache_entries 
                        WHERE expires_at IS NULL AND created_at < ?
                    ''', (age_threshold,))
                    old_count = cursor.rowcount
                    
                    conn.commit()
                
                total_removed = expired_count + old_count
                logger.info(f"Cleaned up {total_removed} expired cache entries")
                return total_removed
                
            except Exception as e:
                logger.error(f"Error cleaning up expired entries: {e}")
                return 0
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit by removing least recently used entries."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Calculate current cache size
                cursor = conn.execute('SELECT SUM(data_size) FROM cache_entries')
                current_size = cursor.fetchone()[0] or 0
                
                if current_size <= self.max_size_bytes:
                    return
                
                # Remove LRU entries until under size limit
                target_size = int(self.max_size_bytes * 0.8)  # Remove to 80% of limit
                
                cursor = conn.execute('''
                    SELECT cache_key, data_size 
                    FROM cache_entries 
                    ORDER BY last_accessed ASC
                ''')
                
                entries_to_remove = []
                size_to_remove = current_size - target_size
                removed_size = 0
                
                for cache_key, data_size in cursor.fetchall():
                    entries_to_remove.append(cache_key)
                    removed_size += data_size
                    
                    if removed_size >= size_to_remove:
                        break
                
                # Remove selected entries
                if entries_to_remove:
                    placeholders = ','.join('?' * len(entries_to_remove))
                    conn.execute(f'''
                        DELETE FROM cache_entries 
                        WHERE cache_key IN ({placeholders})
                    ''', entries_to_remove)
                    conn.commit()
                    
                    logger.info(f"Removed {len(entries_to_remove)} LRU cache entries to enforce size limit")
                
        except Exception as e:
            logger.error(f"Error enforcing cache size limit: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Get entry count and total size
                    cursor = conn.execute('''
                        SELECT COUNT(*), SUM(data_size), AVG(access_count)
                        FROM cache_entries
                    ''')
                    count, total_size, avg_access = cursor.fetchone()
                    
                    # Get hit rate
                    hit_rate = (self._cache_hits / self._total_lookups * 100) if self._total_lookups > 0 else 0
                    
                    return {
                        'total_entries': count or 0,
                        'total_size_mb': (total_size or 0) / (1024 * 1024),
                        'max_size_mb': self.max_size_bytes / (1024 * 1024),
                        'hit_rate_percent': hit_rate,
                        'cache_hits': self._cache_hits,
                        'cache_misses': self._cache_misses,
                        'total_lookups': self._total_lookups,
                        'average_access_count': avg_access or 0
                    }
                    
            except Exception as e:
                logger.error(f"Error getting cache stats: {e}")
                return {}
    
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('DELETE FROM cache_entries')
                    conn.commit()
                
                # Reset statistics
                self._cache_hits = 0
                self._cache_misses = 0
                self._total_lookups = 0
                
                logger.info("Cache cleared successfully")
                
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")