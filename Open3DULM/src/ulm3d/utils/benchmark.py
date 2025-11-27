"""Module for benchmarking ULM pipeline steps."""

import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger


@dataclass
class BenchmarkResult:
    """Store benchmark results for a single step."""
    step_name: str
    execution_times: List[float] = field(default_factory=list)
    
    @property
    def mean_time(self) -> float:
        return np.mean(self.execution_times) if self.execution_times else 0.0
    
    @property
    def std_time(self) -> float:
        return np.std(self.execution_times) if self.execution_times else 0.0
    
    @property
    def min_time(self) -> float:
        return np.min(self.execution_times) if self.execution_times else 0.0
    
    @property
    def max_time(self) -> float:
        return np.max(self.execution_times) if self.execution_times else 0.0
    
    @property
    def total_time(self) -> float:
        return np.sum(self.execution_times) if self.execution_times else 0.0


class BenchmarkManager:
    """Manage benchmarking of ULM pipeline steps."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, BenchmarkResult] = {}
        self.current_step: Optional[str] = None
        self.step_start_time: float = 0.0
        self.total_start_time: Optional[float] = None
        self.total_time: float = 0.0
    
    def start_total(self):
        """Start measuring total execution time."""
        self.total_start_time = time.perf_counter()
    
    def stop_total(self):
        """Stop measuring total execution time."""
        if self.total_start_time is not None:
            self.total_time = time.perf_counter() - self.total_start_time
            logger.info(f"Total execution time: {self.total_time:.2f}s")
    
    @contextmanager
    def measure(self, step_name: str):
        """Context manager to measure execution time of a step."""
        self.current_step = step_name
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.record(step_name, elapsed)
            self.current_step = None
    
    def record(self, step_name: str, execution_time: float):
        """Record execution time for a step."""
        if step_name not in self.results:
            self.results[step_name] = BenchmarkResult(step_name)
        
        self.results[step_name].execution_times.append(execution_time)
        logger.trace(f"[BENCHMARK] {step_name}: {execution_time:.4f}s")
    
    def export_to_csv(self, filename: str = "benchmark_results.csv"):
        """Export benchmark results to CSV file."""
        csv_path = self.output_dir / filename
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'step_name', 
                'n_executions',
                'mean_time_s', 
                'std_time_s',
                'min_time_s',
                'max_time_s',
                'cumulative_time_s',
                'percentage_of_total'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Add total execution time as first row
            writer.writerow({
                'step_name': 'TOTAL_PIPELINE',
                'n_executions': 1,
                'mean_time_s': f"{self.total_time:.6f}",
                'std_time_s': "0.000000",
                'min_time_s': f"{self.total_time:.6f}",
                'max_time_s': f"{self.total_time:.6f}",
                'cumulative_time_s': f"{self.total_time:.6f}",
                'percentage_of_total': "100.00"
            })
            
            # Add individual steps
            for step_name, result in sorted(self.results.items()):
                percentage = (result.total_time / self.total_time * 100) if self.total_time > 0 else 0
                writer.writerow({
                    'step_name': step_name,
                    'n_executions': len(result.execution_times),
                    'mean_time_s': f"{result.mean_time:.6f}",
                    'std_time_s': f"{result.std_time:.6f}",
                    'min_time_s': f"{result.min_time:.6f}",
                    'max_time_s': f"{result.max_time:.6f}",
                    'cumulative_time_s': f"{result.total_time:.6f}",
                    'percentage_of_total': f"{percentage:.2f}"
                })
        
        logger.success(f"Benchmark results exported to {csv_path}")
        return csv_path
    
    def print_summary(self):
        """Print summary of benchmark results."""
        logger.info("\n" + "="*70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*70)
        
        logger.info(f"\nTotal execution time: {self.total_time:.2f}s")
        logger.info("-"*70)
        
        for step_name in sorted(self.results.keys()):
            result = self.results[step_name]
            percentage = (result.total_time / self.total_time * 100) if self.total_time > 0 else 0
            
            logger.info(f"\n{step_name}:")
            logger.info(f"  Executions: {len(result.execution_times)}")
            logger.info(f"  Mean: {result.mean_time:.4f}s ± {result.std_time:.4f}s")
            logger.info(f"  Min/Max: {result.min_time:.4f}s / {result.max_time:.4f}s")
            logger.info(f"  Total: {result.total_time:.4f}s ({percentage:.1f}%)")
        
        logger.info(f"\n{'='*70}\n")
