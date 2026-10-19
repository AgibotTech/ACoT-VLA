#!/usr/bin/env python3
"""Launch a Genie Sim benchmark task with the server-owned RTC stream client (rtc_stream_policy.py).

    python3 run_rtc_stream.py [stream flags] -- <app.py args>

Same launcher as the harness run_rtc.py, with CoRobotPolicy swapped for
RtcStreamPolicy. Everything after `--` goes to geniesim_benchmark/app/app.py
unchanged. Needs ACoT-VLA scripts/serve_rtc_stream.py running with --rtc.clock external;
point --benchmark.infer_host at its port.

    python3 run_rtc_stream.py --trace /workspace/runs/stream.jsonl -- --config ... --benchmark.infer_host=HOST:8001
"""

from __future__ import annotations

import argparse
import os
import pathlib
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _split_argv(argv):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def build_parser():
    """Build the parser for the stream flags (the arguments before ``--``).

    Returns:
        The argparse parser for --pace-hz, --trace, --pipeline and --log-level.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pace-hz", type=float, default=30.0, help="wall-clock act() rate; 0 = unpaced")
    p.add_argument("--trace", default="")
    p.add_argument(
        "--pipeline",
        action="store_true",
        help="send observations without waiting for each reply (sim steps at its own rate)",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main():
    """Configure the stream client, patch CoRobotPolicy on import, and run app.py in-process."""
    mine, theirs = _split_argv(sys.argv[1:])
    a = build_parser().parse_args(mine)

    import logging

    logging.basicConfig(
        level=getattr(logging, a.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # One observation per tick: websockets' DEBUG frame log would dominate the run log.
    logging.getLogger("websockets").setLevel(logging.INFO)

    import rtc_stream_policy

    rtc_stream_policy.CFG = rtc_stream_policy.StreamConfig(pace_hz=a.pace_hz, trace=a.trace, pipeline=a.pipeline)

    # corobotpolicy imports Isaac/omni modules, so it cannot be imported before
    # SimulationApp boots: swap CoRobotPolicy -> RtcStreamPolicy when the benchmark imports it.
    import importlib.abc

    target = "geniesim_benchmark.benchmark.policy.corobotpolicy"

    class _PatchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target_module=None):
            if fullname != target:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target_module) if hasattr(finder, "find_spec") else None
                if spec is not None:
                    break
            else:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module):
                orig_exec(module)
                module.CoRobotPolicy = rtc_stream_policy.build(module.CoRobotPolicy)
                print("[run_rtc_stream] CoRobotPolicy -> RtcStreamPolicy", flush=True)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _PatchFinder())

    import geniesim_benchmark  # app.py boots SimulationApp at import: locate it, don't import it

    app_py = str(pathlib.Path(geniesim_benchmark.__file__).parent / "app" / "app.py")
    print(f"[run_rtc_stream] cfg: {rtc_stream_policy.CFG}", flush=True)
    print(f"[run_rtc_stream] launching {app_py} {' '.join(theirs)}", flush=True)
    sys.argv = [app_py, *theirs]
    runpy.run_path(app_py, run_name="__main__")


if __name__ == "__main__":
    main()
