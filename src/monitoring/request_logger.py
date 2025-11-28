"""Request logging and metrics aggregation."""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RequestLogger:
    """Production request logger with metrics."""
    
    def __init__(self, log_dir: str = "logs/requests"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._validate_directory()
    
    def _validate_directory(self):
        """Validate log directory exists and is writable."""
        if not self.log_dir.exists():
            raise RuntimeError(f"Log directory does not exist: {self.log_dir}")
        if not self.log_dir.is_dir():
            raise RuntimeError(f"Path is not a directory: {self.log_dir}")
    
    def log(
        self,
        question: str,
        answer: str,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log inference request with metrics.
        
        Args:
            question: Input question
            answer: Generated answer
            latency_ms: Request latency in milliseconds
            metadata: Additional metadata
            
        Returns:
            True if logged successfully
        """
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "question": question[:500],
                "answer": answer[:1000],
                "question_length": len(question),
                "answer_length": len(answer),
                "latency_ms": round(latency_ms, 2),
                **(metadata or {})
            }
            
            log_file = self.log_dir / f"requests_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
            return False
    
    def get_stats(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate metrics for specified date.
        
        Args:
            date: Date in YYYYMMDD format, defaults to today
            
        Returns:
            Aggregated metrics
        """
        if date is None:
            date = datetime.utcnow().strftime('%Y%m%d')
        
        log_file = self.log_dir / f"requests_{date}.jsonl"
        
        if not log_file.exists():
            return {"error": "No data for specified date", "date": date}
        
        stats = {
            "date": date,
            "total_requests": 0,
            "avg_latency_ms": 0,
            "min_latency_ms": 0,
            "max_latency_ms": 0,
            "p50_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0,
            "avg_question_length": 0,
            "avg_answer_length": 0,
        }
        
        latencies = []
        q_lengths = []
        a_lengths = []
        
        try:
            with open(log_file, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    stats["total_requests"] += 1
                    latencies.append(entry.get("latency_ms", 0))
                    q_lengths.append(entry.get("question_length", 0))
                    a_lengths.append(entry.get("answer_length", 0))
            
            if stats["total_requests"] > 0:
                stats["avg_latency_ms"] = round(sum(latencies) / len(latencies), 2)
                stats["min_latency_ms"] = round(min(latencies), 2)
                stats["max_latency_ms"] = round(max(latencies), 2)
                
                sorted_latencies = sorted(latencies)
                n = len(sorted_latencies)
                stats["p50_latency_ms"] = round(sorted_latencies[int(n * 0.50)], 2)
                stats["p95_latency_ms"] = round(sorted_latencies[int(n * 0.95)], 2)
                stats["p99_latency_ms"] = round(sorted_latencies[int(n * 0.99)], 2)
                
                stats["avg_question_length"] = round(sum(q_lengths) / len(q_lengths), 1)
                stats["avg_answer_length"] = round(sum(a_lengths) / len(a_lengths), 1)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute stats: {e}")
            return {"error": str(e), "date": date}
