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

"""Torch reference for the chunked_delta_rule kernel: the gated delta rule, token by token."""

import torch


def chunked_delta_rule_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    s0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token recurrence in float64, the semantics of chunked_delta_rule.

    For every v head h (k head h // (Hv // Hk)), from S = s0[h]:
        S_t = diag(exp(g_t)) S_{t-1} + k_t (beta_t (v_t - (diag(exp(g_t)) S_{t-1})^T k_t))^T,  o_t = S_t^T q_t

    Args:
        q, k: [T, Hk, D]; v: [T, Hv, Dv]
        g: [T, Hv, D] (KDA, per-channel log decay) or [T, Hv] (GDN, one log decay per token and head)
        beta: [T, Hv]
        s0: [Hv, D, Dv] initial state

    Returns:
        o: [T, Hv, Dv] float32
        state: [Hv, D, Dv] float32, the final state
    """
    T, Hk, D = k.shape
    Hv = v.shape[1]
    rep = Hv // Hk
    q, k, v, g, beta = (x.double() for x in (q, k, v, g, beta))
    head = torch.arange(Hv) // rep
    S = s0.double().clone()
    o = torch.zeros(T, Hv, v.shape[2], dtype=torch.float64)
    for t in range(T):
        qt, kt = q[t, head], k[t, head]  # [Hv, D]
        e = g[t].exp() if g.ndim == 3 else g[t].exp()[:, None].expand(Hv, D)
        S = e[:, :, None] * S
        delta = beta[t][:, None] * (v[t] - torch.einsum("hdv,hd->hv", S, kt))
        S = S + kt[:, :, None] * delta[:, None, :]
        o[t] = torch.einsum("hdv,hd->hv", S, qt)
    return o.float(), S.float()
