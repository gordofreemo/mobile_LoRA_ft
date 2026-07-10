//
//  SmolLM3.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/7/4.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - NoPE

final class NoPE: Module, OffsetLayer, ArrayOffsetLayer {
    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        return x
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        return x
    }
}

// MARK: - Attention

class SmolLM3Attention: Module {
    let args: SmolLM3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    var rope: RoPELayer

    init(_ args: SmolLM3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: 1.0
        )

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

// MARK: - MLP

class SmolLM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: SmolLM3Configuration) {
        _gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        _down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        _up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = silu(gate(x))
        return down(activation * up(x))
    }
}

class SmolLM3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: SmolLM3Attention
    @ModuleInfo(key: "mlp") var mlp: SmolLM3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: SmolLM3Configuration) {
        _attention.wrappedValue = SmolLM3Attention(args)
        _mlp.wrappedValue = SmolLM3MLP(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

// MARK: - Model

public class SmolLM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [SmolLM3TransformerBlock]
    let norm: RMSNorm

    /// When true, each transformer block's forward is wrapped in a gradient
    /// checkpoint: its activations are NOT retained for backward and are
    /// recomputed on the backward pass instead. Trades compute for a large
    /// reduction in peak activation memory during training. Set this ONLY for
    /// training (it is pure overhead during cached generation). Toggled via
    /// `SmolLM3Model.useGradientCheckpoint`. See the Phase 3 on-device
    /// gradient-checkpointing experiment.
    var useGradientCheckpoint = false

    /// Keeps each step's per-block `CustomFunction` wrappers (and the MLX C
    /// closures they own) alive until the *next* forward begins — by which time
    /// the previous step's backward, which invokes the custom VJP, has long
    /// completed. Without this the wrappers could be released before backward
    /// runs the recompute. Reset at the start of every checkpointed forward.
    private var checkpointRetainer: [Any] = []

    init(_ args: SmolLM3Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in SmolLM3TransformerBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        if useGradientCheckpoint {
            checkpointRetainer.removeAll(keepingCapacity: true)
            for (i, layer) in layers.enumerated() {
                h = checkpointedBlock(layer, h, mask: mask, cache: cache?[i])
            }
        } else {
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }
        }

        return norm(h)
    }

    /// One transformer block's forward, wrapped as a gradient checkpoint.
    ///
    /// Built on the public MLX `CustomFunction` + `vjp` API rather than the raw
    /// `mlx_checkpoint` C symbol (which is unreachable from Swift: `Cmlx` is not
    /// a public product of mlx-swift). The construction is equivalent to
    /// `mx.checkpoint`: the forward runs inside a custom primitive so its
    /// activation tape is dropped, and the VJP re-runs the same forward under
    /// `vjp` to rematerialise activations only for this single block's backward,
    /// then frees them.
    ///
    /// Mirrors mlx_lm's `grad_checkpoint`: the block's trainable (LoRA)
    /// parameters are threaded through the checkpoint as explicit differentiable
    /// inputs (`[h] + params`) so their gradients propagate across the recompute
    /// boundary; captured frozen base weights are constants and need no grad.
    private func checkpointedBlock(
        _ block: SmolLM3TransformerBlock, _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let flat = block.trainableParameters().flattened()
        let keys = flat.map { $0.0 }
        let paramArrays = flat.map { $0.1 }

        // Re-runs the block forward from saved inputs. Used both for the forward
        // pass and for the VJP recompute. `inputs[0]` is `h`; the rest are the
        // trainable parameters in `keys` order.
        func run(_ inputs: [MLXArray]) -> [MLXArray] {
            let h = inputs[0]
            let restored = NestedDictionary<String, MLXArray>.unflattened(
                Array(zip(keys, inputs.dropFirst())))
            block.update(parameters: restored)
            return [block(h, mask: mask, cache: cache)]
        }

        let checkpointed = CustomFunction {
            Forward { inputs in run(inputs) }
            VJP { primals, cotangents in
                vjp(run, primals: primals, cotangents: cotangents).1
            }
        }
        checkpointRetainer.append(checkpointed)

        return checkpointed([x] + paramArrays)[0]
    }
}

public class SmolLM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: SmolLM3ModelInner
    let configuration: SmolLM3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: SmolLM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }

        self.model = SmolLM3ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }

        super.init()

        let identityRope = NoPE()
        for (idx, useRope) in args.noRopeLayers.enumerated() {
            if useRope == 0 && idx < model.layers.count {
                model.layers[idx].attention.rope = identityRope
            }
        }
    }

    /// Enable per-transformer-block gradient checkpointing for subsequent
    /// forward passes. Set `true` ONLY for training: it recomputes each block's
    /// activations on the backward pass to cut peak memory, which is pure
    /// overhead during cached generation. See the Phase 3 on-device
    /// gradient-checkpointing experiment.
    public var useGradientCheckpoint: Bool {
        get { model.useGradientCheckpoint }
        set { model.useGradientCheckpoint = newValue }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        weights = weights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }
}

// MARK: - Configuration

public struct SmolLM3Configuration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = true
    var attentionBias: Bool = false
    var mlpBias: Bool = false

    var noRopeLayerInterval: Int = 4
    var noRopeLayers: [Int] = []

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    public init(
        modelType: String = "smollm3",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        headDimensions: Int? = nil,
        rmsNormEps: Float,
        vocabularySize: Int,
        kvHeads: Int,
        maxPositionEmbeddings: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        noRopeLayerInterval: Int = 4,
        noRopeLayers: [Int]? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.noRopeLayerInterval = noRopeLayerInterval

        if let noRopeLayers = noRopeLayers {
            self.noRopeLayers = noRopeLayers
        } else {
            self.noRopeLayers = (0 ..< hiddenLayers).map { i in
                (i + 1) % noRopeLayerInterval != 0 ? 1 : 0
            }
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case noRopeLayerInterval = "no_rope_layer_interval"
        case noRopeLayers = "no_rope_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "smollm3"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        self.maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        self.ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false

        self.noRopeLayerInterval =
            try container.decodeIfPresent(Int.self, forKey: .noRopeLayerInterval) ?? 4

        if let noRopeLayers = try container.decodeIfPresent([Int].self, forKey: .noRopeLayers) {
            self.noRopeLayers = noRopeLayers
        } else {
            self.noRopeLayers = (0 ..< hiddenLayers).map { i in
                (i + 1) % noRopeLayerInterval != 0 ? 1 : 0
            }
        }
    }
}

// MARK: - LoRA

extension SmolLM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
