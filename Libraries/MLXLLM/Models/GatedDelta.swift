//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Compute G

func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    return decay.asType(a.dtype)
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = state[i];
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?
    let tapeKernel: MLXFast.MLXFastKernel?
    let tapeKernelMasked: MLXFast.MLXFastKernel?
    let kernelsDisabled: Bool

    private init() {
        kernelsDisabled = ProcessInfo.processInfo.environment["DFLASH_DISABLE_GDN_METAL"] == "1"
        kernel = makeGatedDeltaKernel(hasMask: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true)
        tapeKernel = makeGatedDeltaTapeKernel(hasMask: false)
        tapeKernelMasked = makeGatedDeltaTapeKernel(hasMask: true)
    }
}

private func makeGatedDeltaTapeKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;
            auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {
              float delta = 0.0f;
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              }
              if (thread_index_in_simdgroup == 0) {
                tape_[dv_idx] = delta;
              }
              for (int i = 0; i < n_per_t; ++i) {
                state[i] = static_cast<float>(static_cast<InT>(state[i]));
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              tape_ += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = state[i];
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""
    return MLXFast.metalKernel(
        name: "gated_delta_tape\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out", "innovation_tape"],
        source: source
    )
}

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.kernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.kernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, .float32]
    )

    return (outputs[0], outputs[1])
}

func gatedDeltaKernelWithTape(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray, MLXArray)? {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    guard Dk >= 32, Dk % 32 == 0 else {
        return nil
    }

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.tapeKernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.tapeKernel
    }

    guard let kernel = selectedKernel else {
        return nil
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", q.dtype),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape, [B, T, Hv, Dv]],
        outputDTypes: [q.dtype, .float32, .float32]
    )

    return (outputs[0], outputs[1], outputs[2])
}

// MARK: - Ops Fallback

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y.asType(q.dtype), state)
}

func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    var ys = [MLXArray]()
    ys.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = q[0..., t]
        let kT = k[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]
        let maskT = mask == nil ? nil : mask![0..., t]

        let (y, newState) = gatedDeltaStepOps(
            q: qT,
            k: kT,
            v: vT,
            g: gT,
            beta: betaT,
            state: state,
            mask: maskT
        )
        ys.append(y)
        state = newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

func gatedDeltaOpsWithTape(
    q originalQ: MLXArray,
    k originalK: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state initialState: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray, MLXArray) {
    var q = originalQ
    var k = originalK
    let hv = v.dim(2)
    let hk = k.dim(2)
    if hv % hk == 0 {
        let repeatFactor = hv / hk
        if repeatFactor > 1 {
            q = repeated(q, count: repeatFactor, axis: 2)
            k = repeated(k, count: repeatFactor, axis: 2)
        }
    }

    var state = initialState
    var outputs = [MLXArray]()
    var tape = [MLXArray]()
    outputs.reserveCapacity(k.dim(1))
    tape.reserveCapacity(k.dim(1))

    for t in 0 ..< k.dim(1) {
        let oldState = state
        let decay: MLXArray
        if g.ndim == 4 {
            decay = g[0..., t, 0..., .newAxis, 0...]
        } else if g.ndim == 3 {
            decay = g[0..., t, 0..., .newAxis, .newAxis]
        } else {
            fatalError("Unsupported gating shape \(g.shape)")
        }

        let decayedState = state * decay
        let kT = k[0..., t, 0..., .newAxis, 0...]
        let kvMem = (decayedState * kT).sum(axis: -1)
        var delta = (v[0..., t] - kvMem) * beta[0..., t, 0..., .newAxis]
        var newState = decayedState + kT * delta[0..., 0..., 0..., .newAxis]
        var y = (newState * q[0..., t, 0..., .newAxis, 0...]).sum(axis: -1)

        if let mask {
            let maskT = mask[0..., t]
            let stateMask = maskT[0..., .newAxis, .newAxis, .newAxis]
            let outputMask = maskT[0..., .newAxis, .newAxis]
            newState = MLX.where(stateMask, newState, oldState)
            delta = MLX.where(outputMask, delta, MLXArray.zeros(delta.shape, dtype: delta.dtype))
            y = MLX.where(outputMask, y, MLXArray.zeros(y.shape, dtype: y.dtype))
        }

        state = newState
        outputs.append(y)
        tape.append(delta.asType(.float32))
    }

    return (stacked(outputs, axis: 1), state, stacked(tape, axis: 1))
}

// MARK: - Public API

func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b)
    let g = computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    if q.dim(1) == 1,
        !GatedDeltaKernelManager.shared.kernelsDisabled,
        GatedDeltaKernelManager.shared.kernel != nil
    {
        return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}

func gatedDeltaUpdateWithTape(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
    preferMetalTape: Bool = false
) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    let beta = sigmoid(b)
    let g = computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if preferMetalTape,
        !GatedDeltaKernelManager.shared.kernelsDisabled,
        let result = gatedDeltaKernelWithTape(
            q: q,
            k: k,
            v: v,
            g: g,
            beta: beta,
            state: state,
            mask: mask
        )
    {
        return (result.0, result.1, result.2, g)
    }

    let metalResult: (MLXArray, MLXArray)? =
        (!GatedDeltaKernelManager.shared.kernelsDisabled && GatedDeltaKernelManager.shared.kernel != nil)
        ? gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        : nil

    let (out, newState, tape) = gatedDeltaOpsWithTape(
        q: q,
        k: k,
        v: v,
        g: g,
        beta: beta,
        state: state,
        mask: mask
    )
    if let metalResult {
        return (metalResult.0, metalResult.1, tape, g)
    }
    return (out, newState, tape, g)
}
