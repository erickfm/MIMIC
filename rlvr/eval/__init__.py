# Keep this module lightweight. Sub-modules (especially `runner`) pull
# in heavy deps (peppi, duckdb via `rlvr.sampler.mined`) that aren't
# needed for every entry point in this package.
__all__ = ["EVAL_SET_VERSION", "build_eval_set", "run_eval"]


def __getattr__(name):
    if name in ("EVAL_SET_VERSION", "build_eval_set"):
        from rlvr.eval.build_set import EVAL_SET_VERSION, build_eval_set
        return {"EVAL_SET_VERSION": EVAL_SET_VERSION,
                "build_eval_set": build_eval_set}[name]
    if name == "run_eval":
        from rlvr.eval.runner import run_eval
        return run_eval
    raise AttributeError(f"module 'rlvr.eval' has no attribute {name!r}")
