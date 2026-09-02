const layout = @import("layout.zig");
const std = @import("std");
const la = @import("la.zig");
const act = @import("activation.zig");

pub const Gradients = struct {
    weights: layout.WeightsLayout,
    biases: layout.NeuronLayout(true),
    pub fn deinit(self: Gradients) void {
        self.weights.deinit();
        self.biases.deinit();
    }
};

pub const Layer = struct {
    size: usize,
    activation: act.Activation,
};

pub const Network = struct {
    layerCount: usize,
    layers: []const Layer,
    biases: layout.NeuronLayout(true),
    weights: layout.WeightsLayout,
    allocator: std.mem.Allocator,
    learningRate: f32,
    scratchA: la.Matrix(f32),
    scratchB: la.Matrix(f32),

    pub fn init(allocator: std.mem.Allocator, layers: []const Layer, batchSize: usize) !Network {
        var maxLayerSize: usize = 0;
        for (layers) |layer| {
            if (layer.size > maxLayerSize) {
                maxLayerSize = layer.size;
            }
        }

        return Network{
            .layerCount = layers.len,
            .layers = layers,
            .allocator = allocator,
            .biases = try layout.NeuronLayout(true).init(allocator, layers),
            .weights = try layout.WeightsLayout.init(allocator, layers),
            .learningRate = 0.1,
            .scratchA = try la.Matrix(f32).init(allocator, maxLayerSize, batchSize),
            .scratchB = try la.Matrix(f32).init(allocator, maxLayerSize, batchSize),
        };
    }

    pub fn deinit(self: *const Network) void {
        self.biases.deinit();
        self.weights.deinit();
        self.scratchA.deinit();
        self.scratchB.deinit();
    }
    pub fn initActivations(self: Network, batchSize: usize) !layout.BatchedNeuronLayout(false) {
        return try layout.BatchedNeuronLayout(false).init(self.allocator, self.layers, batchSize);
    }
    pub fn initGradients(self: Network) !Gradients {
        return .{
            .weights = try layout.WeightsLayout.init(self.allocator, self.layers),
            .biases = try layout.NeuronLayout(true).init(self.allocator, self.layers),
        };
    }
    fn pow2_f32(x: f32) f32 {
        return std.math.pow(f32, x, 2);
    }
    pub fn calculateError(self: Network, activations: layout.BatchedNeuronLayout(false), correctOutputs: la.Matrix(f32)) f32 {
        const modelOutputs = activations.getLayer(self.layerCount - 1);
        const batchSize = modelOutputs.height();
        const inverseBatchSize = 1.0 / @as(f32, @floatFromInt(batchSize));
        const neuronCount = modelOutputs.width();
        var err: f32 = 0;
        if (neuronCount == 1) {
            // Skip squaring and square rooting for a single neuron
            return inverseBatchSize * la.sum(la.map(la.sub(modelOutputs.getColumnFrog(0), correctOutputs.getColumnFrog(0)), pow2_f32));
        } else {
            for (0..batchSize) |batchIdx| {
                err += std.math.sqrt(la.sum(la.map(la.sub(modelOutputs.getRowFrog(batchIdx), correctOutputs.getRowFrog(batchIdx)), pow2_f32)));
            }
            return inverseBatchSize * err;
        }
    }
    fn mul2(x: f32) f32 {
        return 2 * x;
    }
    pub fn calculateErrorDerivative(self: Network, activations: layout.BatchedNeuronLayout(false), correctOutput: la.Matrix(f32), deriv: la.Matrix(f32)) void {
        const modelOutput = activations.getLayer(self.layerCount - 1);
        la.storeMat(la.mapMat(la.subMat(modelOutput, correctOutput), mul2), deriv);
    }
    pub fn computeActivations(self: Network, inputs: la.Matrix(f32), activations: layout.BatchedNeuronLayout(false)) void {
        // copy inputs to the activations of the first layer
        la.storeMat(inputs, activations.getLayer(0));
        for (1..self.layerCount) |layerIdx| {
            const weights = self.weights.getLayer(layerIdx);
            const prevActivations = activations.getLayer(layerIdx - 1);
            const thisActivations = activations.getLayer(layerIdx);
            const biases = self.biases.getLayer(layerIdx);

            const z = la.addMat(la.matmul(prevActivations, la.transpose(weights)), la.broadcastRows(la.from(biases), prevActivations.height()));
            la.storeMat(la.mapMatRuntime(z, self.layers[layerIdx].activation.activation), thisActivations);
        }
    }
    // Layouts:
    // Weights: B x N_l x N_{l-1}
    // Activations: B x N_l
    // Delta: B x N_l
    // Biases: B x N_l
    pub fn calculateGradients(self: Network, activations: layout.BatchedNeuronLayout(false), gradients: Gradients, correctOutput: la.Matrix(f32), costDerivatives: la.Matrix(f32), batchSize: usize) void {
        // After a lot of thinking, i arrived at this formula:
        // so for weights:
        //                   dz/dw *    delta    *    delta    * ... *     first delta
        // weight derivative:    a * <o'(z) * w> * <o'(z) * w> * ... * <o'(z)> * cost deriv

        // so for biases
        //                   dz/dw *    delta    *    delta    * ... *     first delta
        // bias derivative:      1 * <o'(z) * w> * <o'(z) * w> * ... * <o'(z)> * cost deriv
        // which is the same as delta

        self.calculateErrorDerivative(activations, correctOutput, costDerivatives);
        var layerIdx = self.layerCount - 1;

        var currentDelta = self.scratchA;
        var nextDelta = self.scratchB;

        const inverseBatchSize = 1.0 / @as(f32, @floatFromInt(batchSize));

        while (layerIdx >= 1) : (layerIdx -= 1) {
            // Where to store our delta
            const deltaOut = currentDelta.view(self.layers[layerIdx].size, batchSize);

            if (layerIdx == self.layerCount - 1) {
                // Calculate cost-derivatives * o'(z)
                // For the last layer, all the other terms don't exist yet
                // the activations are stored N_l x B, but the gradients are B x N_l, so the activations require a transpose
                la.storeMat(
                    la.mulMat(la.mapMatRuntime(activations.getLayer(layerIdx), self.layers[layerIdx].activation.activationDerivFromActivation), costDerivatives),
                    deltaOut,
                );
            } else {
                // Where to get the previous delta from
                const deltaIn = nextDelta.view(self.layers[layerIdx + 1].size, batchSize);

                // Get the weights for how this layer affects the following layer.
                const weights = self.weights.getLayer(layerIdx + 1);
                // From the delta of the following layer, we step back by multiplying with the weights over the chain rule,
                // step back again into the activation function by multiplying with its derivative
                la.storeMat(
                    la.mulMat(la.matmul(deltaIn, weights), la.mapMatRuntime(activations.getLayer(layerIdx), self.layers[layerIdx].activation.activationDerivFromActivation)),
                    deltaOut,
                );
            }

            // The biases need to be summed
            for (0..deltaOut.width()) |neuron| {
                gradients.biases.getNeuron(layerIdx, neuron).* = la.sum(deltaOut.getColumnFrog(neuron)) * inverseBatchSize;
            }

            // And now for the weighs we just need to multiply by the activations of the previous layer (this sums over the batches at the same time)
            const prevLayerActivations = activations.getLayer(layerIdx - 1);
            la.storeMat(la.scale(la.matmul(la.transpose(deltaOut), prevLayerActivations), inverseBatchSize), gradients.weights.getLayer(layerIdx));

            // Swap
            const tmp = currentDelta;
            currentDelta = nextDelta;
            nextDelta = tmp;
        }
    }
    pub fn applyBackPropagation(self: Network, gradients: Gradients) void {
        for (1..self.layerCount) |i| {
            for (0..self.layers[i].size) |j| {
                self.biases.getNeuron(i, j).* -= self.learningRate * gradients.biases.getNeuron(i, j).*;
            }
        }
        for (1..self.layerCount) |i| {
            for (0..self.layers[i].size) |j| {
                for (0..self.layers[i - 1].size) |k| {
                    self.weights.getWeight(i, j, k).* -= self.learningRate * gradients.weights.getWeight(i, j, k).*;
                }
            }
        }
    }
};
