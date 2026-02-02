"""Experiment tracking: git info, config, and script content."""

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class GitInfo:
    """Git repository information."""
    commit_hash: str
    branch: str
    is_dirty: bool
    diff: str  # Uncommitted changes


@dataclass
class ExperimentMetadata:
    """Complete experiment metadata."""
    timestamp: str
    git_info: dict[str, Any]
    config: dict[str, Any]
    script_path: str
    script_content: str
    python_version: str
    command_args: list[str]


class ExperimentTracker:
    """
    Tracks experiment metadata for reproducibility.
    
    Logs:
    - Git commit hash and uncommitted changes
    - Full resolved config
    - Script content that was executed
    - Python version and command line args
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_git_info(self) -> GitInfo:
        """Get current git repository state."""
        try:
            # Get commit hash
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            
            # Get branch name
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            
            # Check if working tree is dirty
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            is_dirty = len(status.strip()) > 0
            
            # Get diff of uncommitted changes
            diff = ""
            if is_dirty:
                diff = subprocess.run(
                    ["git", "diff", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            
            return GitInfo(
                commit_hash=commit,
                branch=branch,
                is_dirty=is_dirty,
                diff=diff,
            )
        except subprocess.CalledProcessError:
            return GitInfo(
                commit_hash="unknown",
                branch="unknown",
                is_dirty=True,
                diff="",
            )

    def get_script_content(self) -> tuple[str, str]:
        """Get the path and content of the main script."""
        script_path = sys.argv[0]
        try:
            script_content = Path(script_path).read_text()
        except Exception:
            script_content = ""
        return script_path, script_content

    def log_experiment(self, config: DictConfig) -> ExperimentMetadata:
        """
        Log all experiment metadata.
        
        Args:
            config: Resolved Hydra config.
        
        Returns:
            ExperimentMetadata with all tracked information.
        """
        git_info = self.get_git_info()
        script_path, script_content = self.get_script_content()
        
        metadata = ExperimentMetadata(
            timestamp=datetime.now().isoformat(),
            git_info=asdict(git_info),
            config=OmegaConf.to_container(config, resolve=True),
            script_path=script_path,
            script_content=script_content,
            python_version=sys.version,
            command_args=sys.argv,
        )
        
        # Save to files
        self._save_metadata(metadata)
        
        return metadata

    def _save_metadata(self, metadata: ExperimentMetadata) -> None:
        """Save metadata to output directory."""
        # Save git info
        git_path = self.output_dir / "git_info.json"
        git_path.write_text(json.dumps(metadata.git_info, indent=2))
        
        # Save full metadata
        meta_path = self.output_dir / "experiment_metadata.json"
        meta_dict = {
            "timestamp": metadata.timestamp,
            "git_info": metadata.git_info,
            "config": metadata.config,
            "script_path": metadata.script_path,
            "python_version": metadata.python_version,
            "command_args": metadata.command_args,
        }
        meta_path.write_text(json.dumps(meta_dict, indent=2))
        
        # Save script content separately (can be large)
        if metadata.script_content:
            script_backup = self.output_dir / "script_snapshot.py"
            script_backup.write_text(metadata.script_content)
        
        # Save resolved config as YAML
        config_path = self.output_dir / "config.yaml"
        config_path.write_text(OmegaConf.to_yaml(metadata.config))

    def log_results(
        self,
        results: list[dict[str, Any]],
        filename: str = "results.jsonl",
    ) -> None:
        """
        Log per-example results.
        
        Args:
            results: List of result dicts, one per example.
            filename: Output filename.
        """
        results_path = self.output_dir / filename
        with open(results_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

    def log_metrics(
        self,
        metrics: dict[str, Any],
        filename: str = "metrics.json",
    ) -> None:
        """
        Log aggregated metrics.
        
        Args:
            metrics: Dict of metric name -> value.
            filename: Output filename.
        """
        metrics_path = self.output_dir / filename
        metrics_path.write_text(json.dumps(metrics, indent=2))

    def log_usage(
        self,
        usage: dict[str, int],
        filename: str = "usage.json",
    ) -> None:
        """
        Log token usage statistics.
        
        Args:
            usage: Dict with token counts.
            filename: Output filename.
        """
        usage_path = self.output_dir / filename
        usage_path.write_text(json.dumps(usage, indent=2))


