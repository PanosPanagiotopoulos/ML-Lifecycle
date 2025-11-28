"""MLflow experiment tracking integration."""
import mlflow
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow experiment tracker for training runs."""
    
    def __init__(self, experiment_name: str, tracking_uri: str = "file:./mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run = None
        self._initialize()
    
    def _initialize(self):
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"Experiment tracker initialized: {self.experiment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None) -> Optional[mlflow.ActiveRun]:
        """Start new experiment run."""
        try:
            self.run = mlflow.start_run(run_name=run_name)
            logger.info(f"Started run: {self.run.info.run_id}")
            return self.run
        except Exception as e:
            logger.error(f"Failed to start run: {e}")
            return None
    
    def log_params(self, params: Dict[str, Any]) -> bool:
        """Log training parameters."""
        try:
            safe_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(safe_params)
            return True
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
            return False
    
    def log_metrics(self, metrics: Dict[str, float]) -> bool:
        """Log training metrics."""
        try:
            safe_metrics = {
                k: float(v) for k, v in metrics.items() 
                if v is not None and not (isinstance(v, float) and v != v)
            }
            if safe_metrics:
                mlflow.log_metrics(safe_metrics)
            return True
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
            return False
    
    def log_artifacts(self, artifact_path: str) -> bool:
        """Log model artifacts."""
        try:
            mlflow.log_artifacts(artifact_path)
            logger.info(f"Logged artifacts from {artifact_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to log artifacts: {e}")
            return False
    
    def end_run(self):
        """End current experiment run."""
        if self.run:
            try:
                mlflow.end_run()
                logger.info("Run ended")
            except Exception as e:
                logger.error(f"Failed to end run: {e}")
            finally:
                self.run = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
