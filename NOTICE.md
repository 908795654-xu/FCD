# Attribution and modification notices

FCD contributions: Copyright 2026 FCD contributors, licensed under Apache-2.0.
Third-party copyright and license notices below remain applicable.

## PatchTST

Upstream: https://github.com/yuqinie98/PatchTST
License: Apache-2.0; full upstream license in LICENSES/PatchTST-Apache-2.0.txt.
The PatchTST-derived model/layer implementation is adapted for FCD readouts,
configuration plumbing, and forecasting-head experiments. Extracted frozen-head
code is derived through this FCD implementation, not advertised as unchanged
upstream PatchTST. FCD changes are not endorsed by the upstream authors.

## RevIN

Upstream: https://github.com/ts-kim/RevIN
Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
License: MIT; full text in LICENSES/RevIN-MIT.txt.
The local layer includes FCD-project extensions beyond upstream normalization.

## DLinear / LTSF-Linear

Upstream: https://github.com/cure-lab/LTSF-Linear
Copyright 2022 DLinear Authors. All rights reserved.
License: Apache-2.0; full text in LICENSES/LTSF-Linear-Apache-2.0.txt.
The local DLinear model includes the FCD prediction-head experiment integration.

## Other bundled components

Chronos, Granite TTM, Timer, and standalone TSRBench retain their own LICENSE
and any original NOTICE files in their component directories. This notice does
not replace those files or grant a license to model weights or datasets.
