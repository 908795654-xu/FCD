from pathlib import Path
ROOT = Path(__file__).resolve().parent
ASSETS = ROOT
MODEL_PATHS = {
    ("ttm", 96): ASSETS/"models_cache/ttm-512-96-ft-r2.1",
    ("ttm", 720): ASSETS/"models_cache/ttm-512-720-r2",
    ("chronos", 96): ASSETS/"models_cache/chronos-bolt-mini",
    ("chronos", 720): ASSETS/"models_cache/chronos-bolt-mini",
    ("timer", 96): ASSETS/"models_cache/timer-base-84m",
    ("timer", 720): ASSETS/"models_cache/timer-base-84m",
}
