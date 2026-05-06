import Foundation
import MLX

public final class DFlashRecurrentRollbackCache: MambaCache {
    public private(set) var isArmed = false

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
        _ = prefixLength
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
        guard let snapshot else {
            clearTransients()
            return
        }

        self[0] = snapshot[safe: 0] ?? nil
        self[1] = snapshot[safe: 1] ?? nil

        if let tape, let tapeK, let tapeG, let state = self[1] {
            let acceptedSteps = max(0, acceptedDraftTokens) + 1
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
        }
        return state
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
