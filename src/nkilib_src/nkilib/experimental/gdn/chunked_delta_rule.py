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

"""
Chunked gated delta rule (prefill of Gated DeltaNet and Kimi Delta Attention layers) as one NKI kernel.

From the Kiln project (https://github.com/foxl-ai/kiln, Apache-2.0), where it is the default prefill path for
Gated DeltaNet (GDN) and Kimi Delta Attention (KDA) layers on trn1.

What it computes, for every v head h, from an initial state S [Dk, Dv]:

    S_t = diag(exp(g_t)) S_{t-1} + k_t (beta_t (v_t - (diag(exp(g_t)) S_{t-1})^T k_t))^T
    o_t = S_t^T q_t

with per-channel log decays g [T, Hv, Dk] (KDA, kind=1) or one log decay per token and head
g [T, Hv] (GDN, kind=0). It returns every o_t and the final state. Dk = Dv = 128, Hv a multiple of Hk
(GQA: v head h reads k head h // (Hv // Hk)).

The algorithm is the chunked form of flash-linear-attention (fla, https://github.com/fla-org/flash-linear-attention,
MIT): the intra-chunk products A = beta k k^T and Aqk = q k^T under the decay (fla/ops/kda/chunk_intra.py), the
UT transform T = (I + A)^-1 with w = T (beta k exp(G)) and u = T (beta v) (fla/ops/kda/wy_fast.py,
fla/ops/utils/solve_tril.py), the state recurrence (fla/ops/common/chunk_delta_h.py) and the output
(fla/ops/gla/chunk.py). No fla code is used; what is specific here is how each piece is laid onto the
128 x 128 tensor engine and the vector and scalar engines:

- Chunks of 128 tokens: every chunk matrix is one [128, 128] tile. All matmuls are float32 (an fp32 x fp32
  nc_matmul is exact to ~1.6e-7 relative on trn1), a transpose is a matmul with the identity, exp runs on the
  scalar engine.
- KDA's decayed products A_ij = sum_d (beta k_i)_d k_jd exp(G_id - G_jd) are matmuls over d after factoring
  exp(G_i - G_j) = exp(G_i - G_r) exp(G_r - G_j) through a reference row r. As in fla's safe-gate path, rows
  are cut into sub-chunks of 16 and sub-chunk I uses its first row r = 16 I as the reference for all columns
  j <= 16 I + 15. Inside the diagonal sub-chunk exp(G_r - G_j) <= exp(15 |lower_bound|), so the KDA path
  REQUIRES a lower bound on every log decay (a "safe gate": GLM-5.3-Flash and Kimi K3 use -5, so <= e^75 in
  fp32). KDA without such a bound (for example Kimi-Linear-48B) must not use this kernel.
- GDN's decay is a scalar per (token, head): the products are one matmul each and the decay is applied
  after, as exp(min(G_i - G_j, mask)).
- The UT transform is computed transposed, Y = T^T = (I + A^T)^-1: 8 x 8 diagonal blocks inverted by
  repeated squaring ((I + V)(I + V^2)(I + V^4), exact for nilpotent blocks of 8), then four merge levels
  Y += Y (V o Lo_s) Y (the [[A, 0], [C, D]] block-inverse merge), with block masks as constants:
  18 matmuls per chunk.
- The chunk is applied to the state as one affine map, S' = P S + Q with P = diag(exp(G_L)) - (k exp(G_L - G))^T w
  and Q = (k exp(G_L - G))^T u, and o = (q exp(G) - Aqk w) S + Aqk u. Every term that does not involve S is
  computed before S is known, so the sequential part per chunk is two matmul pairs.
- `units` (chunk, head) units are issued together, every step for each unit in turn, so the in-order engine
  queues hold independent work between dependent steps (2 was the best setting measured on trn1).

IO (all float32, in HBM):
  q, k  [T, Hk, 128]   q already scaled (for example by Dk ** -0.5); l2-normalised q and k are the usual inputs
  v     [T, Hv, 128]
  g     [T, Hv, 128] (KDA) or [T, Hv] (GDN): log decays (<= 0)
  beta  [T, Hv]
  s0    [Hv, 128, 128] initial state
  consts [128, 11, 128] from chunked_delta_rule_constants()
  returns o [T, Hv, 128] and the final state [Hv, 128, 128]
T must be a multiple of 128; pad with beta = g = 0 (padding leaves the state unchanged).

Device time on one trn1 NeuronCore (Neuron SDK 2.32, NKI 0.6.0, neuron-bench nc_latency p50 of the kernel's NEFF),
Hk = Hv = heads, Dk = Dv = 128:

    heads   T      KDA       GDN
    2       512    0.14 ms   0.12 ms
    2       8192   2.00 ms   1.62 ms
    8       2048   2.06 ms   1.63 ms
    8       8192   8.19 ms   6.45 ms
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from ...core.utils.kernel_assert import kernel_assert

L = 128  # tokens per chunk (one tile)
R = 16  # rows per reference sub-chunk (KDA)
B8 = 8  # diagonal blocks of the inverse inverted by squaring
LEVELS = (8, 16, 32, 64)  # merge levels of the inverse
KIND_GDN, KIND_KDA = 0, 1
# Constant tiles (delta_rule_constants()): index of each in the [128, NCST, 128] block.
C_I, C_U, C_SU, C_BL, C_BU, C_LO, C_ONES, C_MZ = 0, 1, 2, 3, 4, 5, 9, 10
NCST = 11
NEG = -1.0e30
F32 = nl.float32


def chunked_delta_rule_constants() -> np.ndarray:
    """[128, NCST, 128] fp32, tile c at [:, c, :] (partition p, column q): I; U [p <= q]; SU [p > q];
    BL [same 8-block, p > q]; BU = BL^T; LO_s for s in LEVELS [same 2s-block, p in its lower half, q in its
    upper half]; ONES; MZ: 0 where p <= q, -1e30 elsewhere."""
    r = np.arange(L)
    p, q = r[:, None], r[None, :]
    tiles = [p == q, p <= q, p > q, (p // B8 == q // B8) & (p > q), (p // B8 == q // B8) & (p < q)]
    for s in LEVELS:
        tiles.append((p // (2 * s) == q // (2 * s)) & (p // s % 2 == 1) & (q // s % 2 == 0))
    tiles.append(np.ones((L, L), dtype=bool))
    out = np.stack([t.astype(np.float32) for t in tiles] + [np.where(p <= q, 0.0, NEG).astype(np.float32)], axis=1)
    return np.ascontiguousarray(out)


def _sb(shape):
    return nl.ndarray(shape, dtype=F32, buffer=nl.sbuf)


def _ps(shape):
    return nl.ndarray(shape, dtype=F32, buffer=nl.psum)


def _mm(dst, st, mv, acc=False):
    nisa.nc_matmul(dst=dst, stationary=st, moving=mv, accumulate=acc)


def _dve(dst, a, b, op):
    nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=op, engine=nisa.vector_engine)


def _act_copy(dst, src, scale=1.0):
    nisa.activation(dst=dst, op=nl.copy, data=src, scale=scale)


def _inverse(NTs, C):
    """Y = T^T = (I - NT)^-1 for each unit's strictly upper NT, each step issued for every unit in turn."""
    n = len(NTs)
    Ns, Nb, NbT, NL = [], [], [], []
    for h in range(n):
        pn = _ps((128, 128))
        _mm(pn, NTs[h], C[:, C_I, :])  # N = NT^T
        ns = _sb((128, 128))
        _act_copy(ns, pn)
        Ns.append(ns)
    for h in range(n):
        nb = _sb((128, 128))
        _dve(nb, Ns[h], C[:, C_BL, :], nl.multiply)
        Nb.append(nb)
        nbt = _sb((128, 128))
        _dve(nbt, NTs[h], C[:, C_BU, :], nl.multiply)
        NbT.append(nbt)
        row = []
        for i in range(len(LEVELS)):
            nl_ = _sb((128, 128))
            _dve(nl_, Ns[h], C[:, C_LO + i, :], nl.multiply)
            row.append(nl_)
        NL.append(row)
    P2, P2T, P4, Y = [], [], [], []
    for h in range(n):
        pp = _ps((128, 2, 128))
        _mm(pp[:, 0, :], NbT[h], Nb[h])  # Nb @ Nb
        _mm(pp[:, 1, :], Nb[h], NbT[h])  # NbT @ NbT
        p2 = _sb((128, 128))
        nisa.tensor_copy(dst=p2, src=pp[:, 0, :], engine=nisa.vector_engine)
        p2t = _sb((128, 128))
        _act_copy(p2t, pp[:, 1, :])
        P2.append(p2)
        P2T.append(p2t)
        y1 = _sb((128, 128))
        _dve(y1, NbT[h], C[:, C_I, :], nl.add)
        Y.append(y1)
    for h in range(n):
        p4p = _ps((128, 128))
        _mm(p4p, P2T[h], P2[h])  # P2 @ P2
        p4 = _sb((128, 128))
        _act_copy(p4, p4p)
        P4.append(p4)
        py = _ps((128, 128))
        _mm(py, P2[h], Y[h])  # P2^T Y
        y2 = _sb((128, 128))
        _dve(y2, Y[h], py, nl.add)
        Y[h] = y2
    for h in range(n):
        py = _ps((128, 128))
        _mm(py, P4[h], Y[h])  # P4^T Y
        y3 = _sb((128, 128))
        _dve(y3, Y[h], py, nl.add)
        Y[h] = y3
    for i in range(len(LEVELS)):
        Xs, Rs = [], []
        for h in range(n):
            px = _ps((128, 128))
            _mm(px, Y[h], C[:, C_I, :])  # X = Y^T
            pr = _ps((128, 128))
            _mm(pr, NL[h][i], Y[h])  # (N o LO)^T Y
            xs = _sb((128, 128))
            _act_copy(xs, px)
            rs = _sb((128, 128))
            nisa.tensor_copy(dst=rs, src=pr, engine=nisa.vector_engine)
            Xs.append(xs)
            Rs.append(rs)
        for h in range(n):
            pw = _ps((128, 128))
            _mm(pw, Xs[h], Rs[h])  # Y R
            yn = _sb((128, 128))
            _dve(yn, Y[h], pw, nl.add)
            Y[h] = yn
    return Y


@nki.jit
def chunked_delta_rule(q, k, v, g, beta, s0, consts, kind: int, units: int = 2):
    """
    Chunked gated delta rule: the prefill of Gated DeltaNet (GDN) and Kimi Delta Attention (KDA) layers.

    See the module docstring for the math and how it is laid onto the engines.

    Dimensions:
        T: tokens, a multiple of 128 (pad with beta = g = 0: padding leaves the state unchanged)
        Hk: q / k heads; Hv: v heads, a multiple of Hk (GQA: v head h reads k head h // (Hv // Hk))
        D = Dv = 128: head dims

    Args:
        q (nl.NkiTensor): [T, Hk, 128] fp32 in HBM, already scaled (for example by D ** -0.5)
        k (nl.NkiTensor): [T, Hk, 128] fp32 in HBM
        v (nl.NkiTensor): [T, Hv, 128] fp32 in HBM
        g (nl.NkiTensor): log decays (<= 0) in HBM, [T, Hv, 128] fp32 for KDA, [T, Hv] fp32 for GDN
        beta (nl.NkiTensor): [T, Hv] fp32 in HBM
        s0 (nl.NkiTensor): [Hv, 128, 128] fp32 in HBM, the initial state
        consts (nl.NkiTensor): [128, 11, 128] fp32 in HBM, from chunked_delta_rule_constants()
        kind (int): KIND_KDA (1) or KIND_GDN (0)
        units (int): (chunk, v head) units whose instructions are interleaved (2 was the best measured on trn1)

    Returns:
        o (nl.NkiTensor): [T, Hv, 128] fp32, every o_t
        S_out (nl.NkiTensor): [Hv, 128, 128] fp32, the final state

    Notes:
        - KDA requires a lower bound on every log decay (the "safe gate", for example -5 * sigmoid(.)),
          see the module docstring. KDA without one must not use this kernel.
        - With an SPMD grid of 2 (LNC 2) each program runs half of the v heads.
    """
    T, Hk, D = q.shape
    Hv, Dv = v.shape[1], v.shape[2]
    kernel_assert(T % 128 == 0, f"chunked_delta_rule needs T a multiple of 128, got T={T}")
    kernel_assert(D == 128 and Dv == 128, f"chunked_delta_rule needs D = Dv = 128, got D={D}, Dv={Dv}")
    kernel_assert(Hv % Hk == 0, f"chunked_delta_rule needs Hv a multiple of Hk, got Hk={Hk}, Hv={Hv}")
    kernel_assert(kind in (KIND_GDN, KIND_KDA), f"kind must be KIND_GDN (0) or KIND_KDA (1), got {kind}")
    rep = Hv // Hk
    NCH = T // 128
    o = nl.ndarray((T, Hv, Dv), dtype=F32, buffer=nl.shared_hbm)
    S_out = nl.ndarray((Hv, D, Dv), dtype=F32, buffer=nl.shared_hbm)
    C = _sb((128, NCST, 128))
    nisa.dma_copy(dst=C, src=consts)
    ident = C[:, C_I, :]
    U = C[:, C_U, :]
    # With an SPMD grid of 2 (LNC=2 on trn2) each program runs its own v heads h0 .. h0 + HP - 1 and writes
    # only their o columns and states. Grid 1: every head.
    npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
    HP = Hv // npg if Hv % npg == 0 else Hv
    h0 = pid * HP if Hv % npg == 0 else 0
    mine = Hv % npg == 0 or pid == 0  # heads that do not split evenly all run on program 0
    St = [None] * Hv
    for h in range(h0, h0 + HP):
        if mine:
            s_init = _sb((128, Dv))
            nisa.dma_copy(dst=s_init, src=s0.ap(pattern=[[Dv, 128], [1, Dv]], offset=h * D * Dv))
            St[h] = s_init
    # Units (chunk c, v head h) in chunk-major order, `units` at a time; the (o, S) chain of a head runs in
    # chunk order.
    NU = NCH * HP if mine else 0
    for u0 in range(0, NU, units):
        heads = []
        t0s = []
        for u in range(u0, min(u0 + units, NU)):
            heads.append(h0 + u % HP)
            t0s.append((u // HP) * 128)
        n = len(heads)
        # Loads (token-major tiles: rows are tokens).
        qt, kt, vt, bt, gt = [], [], [], [], []
        for j in range(n):
            h = heads[j]
            hk = h // rep
            a = _sb((128, D))
            nisa.dma_copy(dst=a, src=q.ap(pattern=[[Hk * D, 128], [1, D]], offset=t0s[j] * Hk * D + hk * D))
            qt.append(a)
            b = _sb((128, D))
            nisa.dma_copy(dst=b, src=k.ap(pattern=[[Hk * D, 128], [1, D]], offset=t0s[j] * Hk * D + hk * D))
            kt.append(b)
            vv = _sb((128, Dv))
            nisa.dma_copy(dst=vv, src=v.ap(pattern=[[Hv * Dv, 128], [1, Dv]], offset=t0s[j] * Hv * Dv + h * Dv))
            vt.append(vv)
            bb = _sb((128, 1))
            nisa.dma_copy(dst=bb, src=beta.ap(pattern=[[Hv, 128], [1, 1]], offset=t0s[j] * Hv + h))
            bt.append(bb)
            if kind == 1:
                gg = _sb((128, D))
                nisa.dma_copy(dst=gg, src=g.ap(pattern=[[Hv * D, 128], [1, D]], offset=t0s[j] * Hv * D + h * D))
            else:
                gg = _sb((128, 1))
                nisa.dma_copy(dst=gg, src=g.ap(pattern=[[Hv, 128], [1, 1]], offset=t0s[j] * Hv + h))
            gt.append(gg)
        # Token-major scaled operands; RHS = [beta v | beta k exp(G)].
        kb, RHS = [], []
        for j in range(n):
            a = _sb((128, D))
            nisa.tensor_scalar(dst=a, data=kt[j], op0=nl.multiply, operand0=bt[j], engine=nisa.vector_engine)
            kb.append(a)
            rh = _sb((128, 2, 128))
            _act_copy(rh[:, 0, :], vt[j], scale=bt[j])
            RHS.append(rh)
        # d-major transposes (and KDA's cumulative decay G^T = g^T U) on the tensor engine.
        qT, kT, kbT, GT = [], [], [], []
        for j in range(n):
            p1 = _ps((128, 4, 128))
            _mm(p1[:, 0, :], qt[j], ident)
            _mm(p1[:, 1, :], kt[j], ident)
            _mm(p1[:, 2, :], kb[j], ident)
            if kind == 1:
                _mm(p1[:, 3, :], gt[j], U)
            a = _sb((128, 128))
            _act_copy(a, p1[:, 0, :])
            qT.append(a)
            b = _sb((128, 128))
            nisa.tensor_copy(dst=b, src=p1[:, 1, :], engine=nisa.vector_engine)
            kT.append(b)
            cc = _sb((128, 128))
            _act_copy(cc, p1[:, 2, :])
            kbT.append(cc)
            if kind == 1:
                gg = _sb((128, 128))
                nisa.tensor_copy(dst=gg, src=p1[:, 3, :], engine=nisa.vector_engine)
                GT.append(gg)
        NT, QkT, qGT, kd, GL = [], [], [], [], []
        if kind == 1:
            for j in range(n):
                # Reference rows: EQ[:, i] = exp(G_i - G_r(i)); kg_I = k^T exp(G_r - G_j), j <= r + 15.
                ngr = _sb((128, 8))
                nisa.tensor_scalar(
                    dst=ngr,
                    data=GT[j].ap(pattern=[[128, 128], [16, 8]], offset=0),
                    op0=nl.multiply,
                    operand0=-1.0,
                    engine=nisa.vector_engine,
                )
                eq = _sb((128, 128))
                for b_ in range(8):
                    nisa.activation(
                        dst=eq[:, b_ * 16 : (b_ + 1) * 16],
                        op=nl.exp,
                        data=GT[j][:, b_ * 16 : (b_ + 1) * 16],
                        bias=ngr[:, b_ : b_ + 1],
                        scale=1.0,
                    )
                kq = _sb((128, 2, 128))  # [kb^T EQ | q^T EQ]
                _dve(kq[:, 0, :], kbT[j], eq, nl.multiply)
                _dve(kq[:, 1, :], qT[j], eq, nl.multiply)
                z = _ps((128, 2, 128))
                for b_ in range(8):
                    w_ = (b_ + 1) * 16
                    ek = _sb((128, 128))
                    nisa.activation(
                        dst=ek[:, 0:w_],
                        op=nl.exp,
                        data=GT[j][:, 0:w_],
                        bias=GT[j][:, b_ * 16 : b_ * 16 + 1],
                        scale=-1.0,
                    )
                    kg = _sb((128, 128))
                    if b_ < 7:
                        nisa.memset(dst=kg[:, w_:128], value=0.0, engine=nisa.gpsimd_engine)
                    _dve(kg[:, 0:w_], kT[j][:, 0:w_], ek[:, 0:w_], nl.multiply)
                    _mm(z[:, :, b_ * 16 : (b_ + 1) * 16], kg, kq[:, :, b_ * 16 : (b_ + 1) * 16])
                qk = _sb((128, 128))
                _dve(qk, z[:, 1, :], U, nl.multiply)
                QkT.append(qk)
                nt = _sb((128, 128))
                _act_copy(nt, z[:, 0, :], scale=-1.0)
                NT.append(nt)
            for j in range(n):
                gam = _sb((128, 128))
                nisa.activation(dst=gam, op=nl.exp, data=GT[j])
                qg = _sb((128, 128))
                _dve(qg, qT[j], gam, nl.multiply)
                qGT.append(qg)
                gl = _sb((128, 1))
                nisa.activation(dst=gl, op=nl.exp, data=GT[j][:, 127:128])
                GL.append(gl)
                p2 = _ps((128, 2, 128))
                _mm(p2[:, 0, :], U, gt[j])  # G token-major
                _mm(p2[:, 1, :], C[:, C_SU, :], gt[j])  # G_L - G
                eg = _sb((128, 2, 128))
                nisa.activation(dst=eg, op=nl.exp, data=p2)
                _dve(RHS[j][:, 1, :], kb[j], eg[:, 0, :], nl.multiply)
                kdd = _sb((128, 128))
                _dve(kdd, kt[j], eg[:, 1, :], nl.multiply)
                kd.append(kdd)
        else:
            for j in range(n):
                gu = _sb((128, 128))
                nisa.tensor_scalar(dst=gu, data=U, op0=nl.multiply, operand0=gt[j], engine=nisa.vector_engine)
                pe = _ps((128, 128))
                _mm(pe, C[:, C_ONES, :], gu)  # [j, i] = G_i
                pg = _ps((128, 1))
                _mm(pg, U, gt[j])  # G_t
                gc = _sb((128, 1))
                nisa.tensor_copy(dst=gc, src=pg, engine=nisa.vector_engine)
                em = _sb((128, 128))
                nisa.scalar_tensor_tensor(
                    dst=em, data=pe, op0=nl.subtract, operand0=gc, op1=nl.minimum, operand1=C[:, C_MZ, :]
                )
                dec = _sb((128, 128))
                nisa.activation(dst=dec, op=nl.exp, data=em)
                gb = _sb((128, 128))
                nisa.activation(dst=gb, op=nl.exp, data=pe)
                GL.append(gb[:, 127:128])
                gm = _sb((128, 1))
                nisa.activation(dst=gm, op=nl.exp, data=gc)
                nisa.tensor_scalar(
                    dst=RHS[j][:, 1, :], data=kb[j], op0=nl.multiply, operand0=gm, engine=nisa.vector_engine
                )
                kdd = _sb((128, 128))
                nisa.tensor_scalar(
                    dst=kdd, data=kt[j], op0=nl.multiply, operand0=dec[:, 127:128], engine=nisa.vector_engine
                )
                kd.append(kdd)
                kq = _sb((128, 2, 128))
                _act_copy(kq[:, 0, :], kbT[j])
                _act_copy(kq[:, 1, :], qT[j])
                z = _ps((128, 2, 128))
                _mm(z, kT[j], kq)  # [k_j . beta k_i | k_j . q_i]
                nt = _sb((128, 128))
                nisa.scalar_tensor_tensor(
                    dst=nt, data=z[:, 0, :], op0=nl.multiply, operand0=-1.0, op1=nl.multiply, operand1=dec
                )
                NT.append(nt)
                qk = _sb((128, 128))
                _dve(qk, z[:, 1, :], dec, nl.multiply)
                QkT.append(qk)
                qg = _sb((128, 128))
                _dve(qg, qT[j], gb, nl.multiply)
                qGT.append(qg)
        Y = _inverse(NT, C)
        # u | w, then every term of the chunk's affine map that does not involve S.
        UW, NPT, MT = [], [], []
        for j in range(n):
            pu = _ps((128, 2, 128))
            _mm(pu, Y[j], RHS[j])  # T [beta v | beta k exp(G)]
            uw = _sb((128, 2, 128))
            _act_copy(uw, pu)
            UW.append(uw)
        for j in range(n):
            pn = _ps((128, 128))
            _mm(pn, UW[j][:, 1, :], kd[j])  # w^T kd [d', d]
            npt = _sb((128, 128))
            nisa.scalar_tensor_tensor(
                dst=npt, data=ident, op0=nl.multiply, operand0=GL[j], op1=nl.subtract, operand1=pn
            )
            NPT.append(npt)
            pm = _ps((128, 128))
            _mm(pm, UW[j][:, 1, :], QkT[j])  # w^T Aqk^T [d, i]
            mt = _sb((128, 128))
            _dve(mt, qGT[j], pm, nl.subtract)
            MT.append(mt)
        # The sequential part: o = M S + Aqk u, S' = P S + Q.
        for j in range(n):
            h = heads[j]
            po = _ps((128, Dv))
            _mm(po, QkT[j], UW[j][:, 0, :])
            _mm(po, MT[j], St[h], acc=True)
            ps_ = _ps((128, Dv))
            _mm(ps_, kd[j], UW[j][:, 0, :])
            _mm(ps_, NPT[j], St[h], acc=True)
            ob = _sb((128, Dv))
            _act_copy(ob, po)
            nisa.dma_copy(dst=o.ap(pattern=[[Hv * Dv, 128], [1, Dv]], offset=t0s[j] * Hv * Dv + h * Dv), src=ob)
            sn = _sb((128, Dv))
            nisa.tensor_copy(dst=sn, src=ps_, engine=nisa.vector_engine)
            St[h] = sn
    for h in range(h0, h0 + HP):
        if mine:
            nisa.dma_copy(dst=S_out.ap(pattern=[[Dv, 128], [1, Dv]], offset=h * D * Dv), src=St[h])
    if npg > 1:  # both programs' heads written before either ends
        nisa.core_barrier(data=o, cores=(0, 1))
        nisa.core_barrier(data=S_out, cores=(0, 1))
    return o, S_out
