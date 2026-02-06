import argparse
from dataclasses import dataclass
# from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import numpy as np


# ------------------------------------------------------------------
# Config object passed to the simulator
# ------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Batch/study configuration: trial selection, execution controls, and stage toggles."""
    trials_file: str
    trial_start: int = 0
    trial_end: Optional[int] = None
    trial_range: Optional[str] = None

    single_thread: bool = False
    reduced: bool = False
    max_workers: Optional[int] = None

    profile_cpu: bool = False
    profile_mem: bool = False

    log_level: str = "INFO"
    quiet: bool = False

    force_precompute: bool = False
    force_simulate: bool = False
    force_postprocess: bool = False
    force_all: bool = False

    only_precompute: bool = False
    only_simulate: bool = False
    only_postprocess: bool = False

    def normalize_and_validate(self) -> "SimulationConfig":
        if self.trial_range:
            start_s, end_s = self.trial_range.split(":")
            self.trial_start = int(start_s)
            self.trial_end = int(end_s)

        if self.force_all:
            self.force_precompute = True
            self.force_simulate = True
            self.force_postprocess = True

        if sum((self.only_precompute, self.only_simulate, self.only_postprocess)) > 1:
            raise ValueError("Only one --only-* flag can be used at a time.")
        return self
    
# ------------------------------------------------------------------
# Config object passed to each simulation run
# ------------------------------------------------------------------

@dataclass
class RunConfig:
    """Only what a single trial needs to build/run mission specs."""
    duration: float
    step_size: float
    base_path: str = "./results"

    mission_specs_template: Dict[str, Any] = field(default_factory=dict)
    spacecraft_specs_template: Dict[str, Any] = field(default_factory=dict)
    instrument_specs: Dict[str, Any] = field(default_factory=dict)
    ground_operator_specs_template: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------

def parse_study_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description="Run batch satellite mission simulations"
    )

    # --------------------------------------------------------------
    # Trial inputs
    # --------------------------------------------------------------
    parser.add_argument("-t", "--trials", required=True,
                        help="CSV file containing trial scenarios")

    parser.add_argument("-i", "--trial-start", type=int, default=0,
                        help="Lower trial index bound")

    parser.add_argument("-j", "--trial-end", type=int, default=np.Inf,
                        help="Upper trial index bound")

    parser.add_argument("--trial-range",
                        help="Trial range in format START:END (overrides start/end)")

    # --------------------------------------------------------------
    # Profiling
    # --------------------------------------------------------------
    parser.add_argument("-p", "--profile-cpu", action="store_true",
                        help="Enable CPU profiling (cProfile)")

    parser.add_argument("-m", "--profile-mem", action="store_true",
                        help="Enable memory profiling (tracemalloc)")

    # --------------------------------------------------------------
    # Execution mode
    # --------------------------------------------------------------
    parser.add_argument("-1", "--single-thread", action="store_true",
                        help="Run simulation single-threaded")

    parser.add_argument("-r", "--reduced", action="store_true",
                        help="Run reduced-complexity debug scenarios")

    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------
    parser.add_argument("-l", "--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("-q", "--quiet", action="store_true",
                    help="Disable progress bars and console output (batch-safe mode)")

    # --------------------------------------------------------------
    # Stage overwrite controls
    # --------------------------------------------------------------
    parser.add_argument("-F", "--force-all", action="store_true",
                        help="Force rerun of all stages")

    parser.add_argument("--force-precompute", action="store_true")
    parser.add_argument("--force-simulate", action="store_true")
    parser.add_argument("--force-postprocess", action="store_true")

    # --------------------------------------------------------------
    # Stage isolation (debugging)
    # --------------------------------------------------------------
    parser.add_argument("--only-precompute", action="store_true")
    parser.add_argument("--only-simulate", action="store_true")
    parser.add_argument("--only-postprocess", action="store_true")

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Trial range resolution
    # --------------------------------------------------------------
    trial_start = args.trial_start
    trial_end = args.trial_end

    if args.trial_range:
        try:
            start, end = args.trial_range.split(":")
            trial_start = int(start)
            trial_end = int(end)
        except Exception:
            raise ValueError("Invalid --trial-range format. Use START:END")

    # --------------------------------------------------------------
    # Force logic
    # --------------------------------------------------------------
    force_precompute = args.force_precompute
    force_simulate = args.force_simulate
    force_postprocess = args.force_postprocess

    if args.force_all:
        force_precompute = True
        force_simulate = True
        force_postprocess = True

    # --------------------------------------------------------------
    # Only-stage validation
    # --------------------------------------------------------------
    only_flags = [
        args.only_precompute,
        args.only_simulate,
        args.only_postprocess,
    ]

    if sum(only_flags) > 1:
        raise ValueError("Only one --only-* flag can be used at a time.")

    # --------------------------------------------------------------
    # Build config
    # --------------------------------------------------------------

    return SimulationConfig(
        trials_file=args.trials,
        trial_start=trial_start,
        trial_end=trial_end,

        profile_cpu=args.profile_cpu,
        profile_mem=args.profile_mem,
        single_thread=args.single_thread,
        reduced=args.reduced,

        log_level=args.log_level,
        quiet=args.quiet,

        force_precompute=force_precompute,
        force_simulate=force_simulate,
        force_postprocess=force_postprocess,

        only_precompute=args.only_precompute,
        only_simulate=args.only_simulate,
        only_postprocess=args.only_postprocess,
    )