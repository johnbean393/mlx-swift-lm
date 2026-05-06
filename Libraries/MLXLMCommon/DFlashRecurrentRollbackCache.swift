import Foundation
import MLX

private func makeDFlashTapeReplayKernel(vectorized: Bool) -> MLXFast.MLXFastKernel {
    let gComment: String
    let gSetup: String
    let gAccess: String
    let gAdvance: String
    if vectorized {
        gComment = "// g: [B, T, Hv, Dk]"
        gSetup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        gAccess = "g_[s_idx]"
        gAdvance = "g_ += Hv * Dk;"
    } else {
        gComment = "// g: [B, T, Hv]"
        gSetup = "auto g_ = g + b_idx * T * Hv;"
        gAccess = "g_[hv_idx]"
        gAdvance = "g_ += Hv;"
    }

    let suffix = vectorized ? "_vec" : ""
    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // tape: [B, T, Hv, Dv]
            auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;

            // k: [B, T, Hk, Dk]
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            \(gComment)
            \(gSetup)

            for (int t = 0; t < T; ++t) {
              auto delta = static_cast<float>(tape_[dv_idx]);
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * \(gAccess);
                state[i] = state[i] + k_[s_idx] * delta;
              }
              for (int i = 0; i < n_per_t; ++i) {
                state[i] = static_cast<float>(static_cast<InT>(state[i]));
              }
              tape_ += Hv * Dv;
              k_ += Hk * Dk;
              \(gAdvance)
            }

            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """

    return MLXFast.metalKernel(
        name: "dflash_tape_replay\(suffix)",
        inputNames: ["tape", "k", "g", "state_in", "T"],
        outputNames: ["state_out"],
        source: source
    )
}

private final class DFlashTapeReplayKernelManager: Sendable {
    static let shared = DFlashTapeReplayKernelManager()

    let kernel: MLXFast.MLXFastKernel
    let vectorizedKernel: MLXFast.MLXFastKernel
    let kernelsDisabled: Bool

    private init() {
        kernel = makeDFlashTapeReplayKernel(vectorized: false)
        vectorizedKernel = makeDFlashTapeReplayKernel(vectorized: true)
        kernelsDisabled = ProcessInfo.processInfo.environment["DFLASH_DISABLE_GDN_METAL"] == "1"
    }
}

public final class DFlashRecurrentRollbackCache: MambaCache {
    public private(set) var isArmed = false
    public private(set) var armedPrefixLength = 0

    private let convKernelSize: Int
    private var tape: MLXArray?
    private var tapeK: MLXArray?
    private var tapeG: MLXArray?
    private var tapeQKV: MLXArray?
    private var snapshot: [MLXArray?]?

    public init(convKernelSize: Int = 4, leftPadding: [Int]? = nil) {
        self.convKernelSize = convKernelSize
        super.init(leftPadding: leftPadding)
    }

    public func armRollback(prefixLength: Int = 0) {
        armedPrefixLength = prefixLength
        isArmed = true
        tape = nil
        tapeK = nil
        tapeG = nil
        tapeQKV = nil
        snapshot = [self[0], self[1]]
    }

    public func recordTape(tape: MLXArray, k: MLXArray, g: MLXArray, qkv: MLXArray) {
        self.tape = tape.asType(.float32)
        self.tapeK = k
        self.tapeG = g
        self.tapeQKV = qkv
    }

    public func rollback(acceptedDraftTokens: Int) {
        rollback(committedSteps: max(0, acceptedDraftTokens) + 1)
    }

    public func rollback(committedSteps: Int) {
        guard let snapshot else {
            clearTransients()
            return
        }

        self[0] = snapshot[safe: 0] ?? nil
        self[1] = snapshot[safe: 1] ?? nil

        if let tape, let tapeK, let tapeG, let state = self[1] {
            let acceptedSteps = max(0, committedSteps)
            if acceptedSteps > 0 {
                self[1] = DFlashRecurrentRollbackCache.replay(
                    tape: tape[0..., 0 ..< min(acceptedSteps, tape.dim(1)), 0..., 0...],
                    k: tapeK[0..., 0 ..< min(acceptedSteps, tapeK.dim(1)), 0..., 0...],
                    g: tapeG[0..., 0 ..< min(acceptedSteps, tapeG.dim(1)), 0...],
                    state: state
                )
                self[0] = rebuildConvState(acceptedSteps: acceptedSteps)
            }
        }

        clearTransients()
    }

    public func clearTransients() {
        isArmed = false
        armedPrefixLength = 0
        tape = nil
        tapeK = nil
        tapeG = nil
        tapeQKV = nil
        snapshot = nil
    }

    public override func copy() -> any KVCache {
        let new = DFlashRecurrentRollbackCache(convKernelSize: convKernelSize)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        return new
    }

    private func rebuildConvState(acceptedSteps: Int) -> MLXArray? {
        guard let tapeQKV else {
            return self[0]
        }
        let keep = convKernelSize - 1
        guard keep > 0 else {
            return nil
        }

        let savedConvState = snapshot.flatMap { $0[safe: 0] } ?? nil
        let prefix = savedConvState ?? MLXArray.zeros(
            [tapeQKV.dim(0), keep, tapeQKV.dim(-1)],
            dtype: tapeQKV.dtype
        )
        let convInput = concatenated([prefix, tapeQKV], axis: 1)
        let start = min(max(0, acceptedSteps), convInput.dim(1))
        let end = min(start + keep, convInput.dim(1))
        return convInput[0..., start ..< end, 0...]
    }

    private static func replay(
        tape: MLXArray,
        k originalK: MLXArray,
        g: MLXArray,
        state initialState: MLXArray
    ) -> MLXArray {
        if let replayed = replayWithMetal(tape: tape, k: originalK, g: g, state: initialState) {
            return replayed
        }

        var k = originalK
        let hv = tape.dim(2)
        let hk = k.dim(2)
        if hv % hk == 0 {
            let repeatFactor = hv / hk
            if repeatFactor > 1 {
                k = repeated(k, count: repeatFactor, axis: 2)
            }
        }

        var state = initialState
        for t in 0 ..< tape.dim(1) {
            let decay: MLXArray
            if g.ndim == 4 {
                decay = g[0..., t, 0..., .newAxis, 0...]
            } else {
                decay = g[0..., t, 0..., .newAxis, .newAxis]
            }
            let delta = tape[0..., t, 0..., 0..., .newAxis]
            let kT = k[0..., t, 0..., .newAxis, 0...]
            state = state * decay
            state = state + delta * kT
            state = state.asType(initialState.dtype)
        }
        return state
    }

    private static func replayWithMetal(
        tape: MLXArray,
        k: MLXArray,
        g: MLXArray,
        state: MLXArray
    ) -> MLXArray? {
        guard !DFlashTapeReplayKernelManager.shared.kernelsDisabled else {
            return nil
        }
        let batchSize = k.dim(0)
        let steps = k.dim(1)
        let hk = k.dim(2)
        let dk = k.dim(3)
        let hv = tape.dim(2)
        let dv = tape.dim(3)
        guard steps > 0, dk >= 32, dk % 32 == 0, hv % hk == 0 else {
            return nil
        }

        let selectedKernel: MLXFast.MLXFastKernel
        if g.ndim == 4 {
            selectedKernel = DFlashTapeReplayKernelManager.shared.vectorizedKernel
        } else if g.ndim == 3 {
            selectedKernel = DFlashTapeReplayKernelManager.shared.kernel
        } else {
            return nil
        }

        let outputs = selectedKernel(
            [tape, k, g, state, MLXArray(steps)],
            template: [
                ("InT", state.dtype),
                ("Dk", dk),
                ("Dv", dv),
                ("Hk", hk),
                ("Hv", hv),
            ],
            grid: (32, dv, batchSize * hv),
            threadGroup: (32, 4, 1),
            outputShapes: [state.shape],
            outputDTypes: [state.dtype]
        )
        return outputs[0]
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
