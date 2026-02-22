"""
QuantaAlpha custom workspace.

Overrides rdagent QlibFBWorkspace:
- Injects project-level factor_template YAML over rdagent defaults.
- Overrides execute() to use the project's own QlibLocalEnv instead of rdagent's
  QlibCondaEnv.  rdagent's QlibCondaEnv has two macOS-incompatible behaviours:
    1. It calls `conda create` via /bin/sh which can't find the `conda` shell
       function (conda is initialized only for interactive shells).
    2. It wraps every command in `timeout --kill-after=10 ...` which doesn't
       exist on macOS without installing GNU coreutils.
  The project's QlibLocalEnv avoids both issues:
    - It inherits os.environ (full PATH with conda, qrun, etc.).
    - It uses Python's subprocess timeout, not the shell `timeout` binary.
- Inits an empty git repo in the workspace to suppress qlib recorder git output.
"""

import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace as _RdagentQlibFBWorkspace

_CUSTOM_TEMPLATE_DIR = Path(__file__).resolve().parent / "factor_template"

# Default timeout (seconds) used when config is not available.
_DEFAULT_BACKTEST_TIMEOUT = 800


class QlibFBWorkspace(_RdagentQlibFBWorkspace):
    """
    Override rdagent QlibFBWorkspace: inject project factor_template/ YAML over defaults;
    init empty git repo in workspace to avoid qlib recorder git help output.
    """

    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(template_folder_path, *args, **kwargs)
        if _CUSTOM_TEMPLATE_DIR.exists():
            self.inject_code_from_folder(_CUSTOM_TEMPLATE_DIR)
            logger.info(f"Overrode rdagent default config with project template: {_CUSTOM_TEMPLATE_DIR}")

    def before_execute(self) -> None:
        """Init empty git repo in workspace to suppress qlib recorder git warnings."""
        super().before_execute()
        git_dir = self.workspace_path / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=str(self.workspace_path),
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # execute() — replaces rdagent's QlibCondaEnv with QlibLocalEnv
    # ------------------------------------------------------------------
    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict = {},
                *args: Any, **kwargs: Any):
        """Run qlib backtest locally, bypassing rdagent's conda+timeout path.

        Uses QlibLocalEnv which:
        - Inherits the full os.environ (conda bin, qrun, etc. are all reachable).
        - Uses Python's subprocess timeout — no need for GNU `timeout` binary.

        Returns the same (metrics_series, log_str) tuple as rdagent's execute().
        """
        # Lazy import to avoid circular deps and only pay the import cost here.
        from quantaalpha.utils.env import QlibLocalEnv

        # Respect backtest.timeout from experiment config when available.
        timeout = _DEFAULT_BACKTEST_TIMEOUT
        try:
            import os, yaml
            cfg_path = Path(os.environ.get("CONFIG_PATH", "configs/experiment.yaml"))
            if not cfg_path.is_absolute():
                cfg_path = Path(__file__).resolve().parents[3] / cfg_path
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                timeout = cfg.get("backtest", {}).get("timeout", _DEFAULT_BACKTEST_TIMEOUT)
        except Exception:
            pass  # fall back to default

        local_env = QlibLocalEnv(timeout=timeout)
        local_env.prepare()

        workspace = str(self.workspace_path)

        # --- Step 1: run qlib backtest ---
        execute_qlib_log = ""
        try:
            execute_qlib_log = local_env.run(
                entry=f"qrun {qlib_config_name}",
                local_path=workspace,
                env=run_env,
            )
        except RuntimeError as exc:
            execute_qlib_log = str(exc)
            logger.error(f"qrun failed: {exc}")

        logger.info(f"Qlib execute log (tail): {execute_qlib_log[-500:] if execute_qlib_log else '(empty)'}")

        # --- Step 2: extract results ---
        try:
            local_env.run(
                entry="python read_exp_res.py",
                local_path=workspace,
                env=run_env,
            )
        except RuntimeError as exc:
            logger.error(f"read_exp_res.py failed: {exc}")

        # --- Step 3: read result files (same as rdagent's execute) ---
        ret_path = self.workspace_path / "ret.pkl"
        if not ret_path.exists():
            logger.error("No result file found.")
            return None, execute_qlib_log

        ret_df = pd.read_pickle(ret_path)
        logger.info(f"Quantitative Backtesting Chart loaded: {ret_df.shape if hasattr(ret_df, 'shape') else 'ok'}")

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            short_log = "\n".join(matches) if matches else execute_qlib_log
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], short_log

        logger.error(f"File {qlib_res_path} does not exist.")
        return None, execute_qlib_log
