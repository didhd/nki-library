# Copyright (c) 2026 Sanghwa Na
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the chunked_delta_rule kernel (GDN and KDA prefill)."""

from typing import final

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.gdn.chunked_delta_rule import (
    KIND_GDN,
    KIND_KDA,
    chunked_delta_rule,
    chunked_delta_rule_constants,
)
from nkilib_src.nkilib.experimental.gdn.chunked_delta_rule_torch import chunked_delta_rule_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_inputs(seq_len: int, k_heads: int, v_heads: int, kind: int, seed: int = 0, lower_bound: float = -5.0):
    """Inputs as a GDN / KDA mixer builds them: l2-normalised q (scaled by D ** -0.5) and k, beta in [0, 1),
    KDA gates lower_bound * sigmoid(.) (the safe gate), GDN gates -exp(A) * softplus(.), a random initial state."""
    D = 128
    rng = np.random.default_rng(seed)

    def l2(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    k = l2(rng.standard_normal((seq_len, k_heads, D))).astype(np.float32)
    beta = rng.random((seq_len, v_heads)).astype(np.float32)
    q = (l2(rng.standard_normal((seq_len, k_heads, D))) * D**-0.5).astype(np.float32)
    v = rng.standard_normal((seq_len, v_heads, D)).astype(np.float32)
    if kind == KIND_KDA:
        g = lower_bound / (1.0 + np.exp(-(rng.standard_normal((seq_len, v_heads, D)) * 2)))
    else:
        g = -np.exp(rng.standard_normal(v_heads))[None, :] * np.log1p(np.exp(rng.standard_normal((seq_len, v_heads))))
    s0 = (rng.standard_normal((v_heads, D, D)) * 0.1).astype(np.float32)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g.astype(np.float32),
        "beta": beta,
        "s0": s0,
        "consts": chunked_delta_rule_constants(),
        "kind": kind,
        "units": 2,
    }


@torch_ref_wrapper
def _torch_ref(q, k, v, g, beta, s0, consts, kind, units):
    import torch

    o, state = chunked_delta_rule_torch_ref(*(torch.from_numpy(np.asarray(x)) for x in (q, k, v, g, beta, s0)))
    return {"o": o.numpy(), "s_out": state.numpy()}


@final
@pytest_test_metadata(name="Chunked Delta Rule")
@pytest_marks(["attention", "gdn", "kda"])
class TestChunkedDeltaRule:
    params = "lnc, seq_len, k_heads, v_heads, kind"
    _ABBREVS = {"lnc": "lnc", "seq_len": "s", "k_heads": "hk", "v_heads": "hv", "kind": "kind"}

    test_cases = [
        pytest.param(1, 128, 1, 1, KIND_GDN, marks=pytest.mark.fast),
        pytest.param(1, 256, 2, 2, KIND_KDA, marks=pytest.mark.fast),
        (1, 512, 2, 4, KIND_GDN),  # GQA
        (1, 512, 4, 4, KIND_KDA),
        (1, 2048, 4, 8, KIND_GDN),
        (1, 2048, 4, 4, KIND_KDA),
        (2, 512, 2, 4, KIND_GDN),  # SPMD grid of 2: half of the v heads per program (LNC 2)
        (2, 512, 4, 4, KIND_KDA),
    ]

    @pytest_parametrize(params, test_cases, abbrevs=_ABBREVS)
    def test_chunked_delta_rule(
        self, test_manager: Orchestrator, platform_target: Platforms, lnc, seq_len, k_heads, v_heads, kind
    ):
        def input_generator(test_config, input_tensor_def=None):
            return _generate_inputs(seq_len, k_heads, v_heads, kind, seed=seq_len + kind)

        def output_tensors(kernel_input):
            T, Hv, Dv = kernel_input["v"].shape
            return {
                "o": np.zeros((T, Hv, Dv), dtype=np.float32),
                "s_out": np.zeros((Hv, kernel_input["k"].shape[2], Dv), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=chunked_delta_rule,
            torch_ref=_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            rtol=2e-3,
            atol=1e-4,
        )
