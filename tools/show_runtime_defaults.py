"""Print environment and complete optimizer defaults; no model or training."""
import importlib.metadata
import json
import platform


def main():
    import torch
    parameter = torch.nn.Parameter(torch.zeros(1))
    packages = {}
    for name in ('torch', 'numpy', 'pandas', 'transformers', 'granite-tsfm', 'chronos-forecasting'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = dict(python=platform.python_version(), packages=packages,
                  cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                  cudnn_version=torch.backends.cudnn.version(),
                  note='Library defaults, not per-run overrides. CPU scalar only; no step/forward/backward.',
                  Adam=torch.optim.Adam([parameter]).defaults,
                  AdamW=torch.optim.AdamW([parameter]).defaults)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
