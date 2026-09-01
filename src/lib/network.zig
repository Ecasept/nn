const layout = @import("layout.zig");
const std = @import("std");
const la = @import("la.zig");

const Gradients = struct {
    weights: layout.WeightsLayout,
    biases: layout.NeuronLayout(true),
    pub fn deinit(self: Gradients) void {
        self.weights.deinit();
        self.biases.deinit();
    }
};

const Network = struct {
    layerCount: usize,
    layers: []const usize,
    biases: layout.NeuronLayout(true),
    weights: layout.WeightsLayout,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, layers: []const usize) !Network {
        return Network{ .layerCount = layers.len, .layers = layers, .allocator = allocator, .biases = try layout.NeuronLayout(true).init(allocator, layers), .weights = try layout.WeightsLayout.init(allocator, layers) };
    }

    pub fn deinit(self: *const Network) void {
        self.biases.deinit();
        self.weights.deinit();
    }
    pub fn initActivations(self: Network) !layout.NeuronLayout(false) {
        return try layout.NeuronLayout(false).init(self.allocator, self.layers);
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
    pub fn calculateError(self: Network, activations: layout.NeuronLayout(false), correctOutputs: la.Matrix(f32), errors: []f32) void {
        const modelOutputs = activations.getLayer(self.layerCount - 1);
        if (modelOutputs.height() == 1) {
            la.store(
                la.apply(
                    la.sub(la.from(modelOutputs[0]), la.from(correctOutputs[0])),
                    self.pow2_f32
                ),
                errors
            );
        } else {
            for (0..modelOutputs.height()) |i| {
                errors[i] = la.sum(
                    la.map(
                        la.sub(
                            la.matColumn(modelOutputs, i),
                            la.matColumn(correctOutputs, i)
                        ),
                        self.pow2_f32
                    ),
                );
            }
        }
    }
    fn mul2(x: f32) f32 {
        return 2 * x;
    }
    pub fn calculateErrorDerivative(self: Network, activations: layout.NeuronLayout(false), correctOutput: []const f32, deriv: la.Matrix(f32)) void {
        const modelOutput = activations.getLayer(self.layerCount - 1);
        la.store(
            la.map(
                la.sub(la.from(modelOutput), la.from(correctOutput)),
                mul2
            ),
            deriv
        );
    }
    pub fn calculateGradients(self: Network, activations: layout.NeuronLayout(false), gradients: Gradients, correctOutput: la.Matrix(f32), costDerivatives: la.Matrix(f32)) void {
        self.calculateErrorDerivative(activations, correctOutput, costDerivatives);
        var layerIdx = self.layerCount;
        while (layerIdx > 1) {
            layerIdx -= 1;
            const biasGradients = gradients.biases.getLayer(layerIdx);
            if (layerIdx != self.layerCount - 1) {
                const nextLayerBiasDerivatives = gradients.biases.getLayer(layerIdx + 1);
                // We want to get dC/db
                // if we change the bias by db, the activation changes by sigma'(activation)
                // so da/db = o'(z)

                // and our final result dC/db = dC/da * da/db = dC/da * o'(z)
                
                // and dC/da = dC/da_ * da_/dz * dz/da = ... * o'(z) * w
                
                // so for weights:
                //                   dz/dw * da/dz * next neuron * next neuron * ....    
                // weight derivative:    a * o'(z) * <w * o'(z)> * <w * o'(z)> * ....
                
                // so for biases
                //                   dz/db * da/dz * next neuron * next neuron * ....
                // bias derivative:      1 * o'(z) * <w * o'(z)> * <w * o'(z)> * ....

                // but at every step the neuron affects all neurons of the next layer,
                // there is no single next neuron. so we need to sum up all the next layers

                // so to get the bias derivative:
                // multiply the results of the next neuron with the weight from the current neuron, sum up, and times activation derivative
                // and then vectorize
                const nextLayerWeights = self.weights.getLayer(layerIdx + 1);

                // each row of the bias derivatives contains a single instance from the batch
                // each column of the weights contains a single neuron
                // so after multiplying the bias derivatives with the weights,
                // we get a matrix where each row is a single instance,
                // however, the activation derivative is stored as a column per instance,
                // but needs to be multiplied elementwise, so it needs a transpose
                la.store(
                    la.mulMat(
                        la.matmul(nextLayerBiasDerivatives, nextLayerWeights),
                        la.map(
                            la.transpose(activations.getLayer(layerIdx)),
                            activationDeriv
                        )
                    ),
                    biasGradients,
                );
            } else {
                // For the last layer, all the other terms don't exist yet
                la.store(
                    la.mulMat(
                        la.map(
                            activations.getLayer(layerIdx),
                            activationDeriv
                        ),
                        costDerivatives
                    )
                    biasGradients,
                )
            }

            // And now for the weighs we just need to multiply by the activations of the previous layer
            const prevLayerActivations = activations.getLayer(layerIdx - 1);
            la.store(
                la.m(biasGradients)
            )
            // so the first layer of the output matrix for the weights
            // contains the derivatives for all the weights connected to the first neuron
            // the second layer contains the derivatives for all the weights connected to the second neuron, and so on
            // so the upper left corner contains the derivative of the first weight connected to the first neuron,
            // so this needs to be calculated as the gradient of the leftmost neuron in the gradients times the top-left most activation
            // for the second neuron weight connected to the first neuron, it is the leftmost gradient times the top-second-left-most activation,
            // so to calculate the weight of the j-th neuron connected to the i-th neuron before it,
            // we need to multiply the j-th gradient with the i-th activation row.
            // ie the activations
            la.columnVecMulRowVec(biasGradients, prevLayerActivations, gradients.weights.getLayer(layerIdx));
        }
    }
    pub fn applyBackPropagation(self: Network, gradients: Gradients) void {
        for (1..self.layerCount) |i| {
            for (0..self.layers[i]) |j| {
                self.biases.getNeuron(i, j).* -= LEARNING_RATE * gradients.biases.getNeuron(i, j).*;
            }
        }
        for (1..self.layerCount) |i| {
            for (0..self.layers[i]) |j| {
                for (0..self.layers[i - 1]) |k| {
                    self.weights.getWeight(i, j, k).* -= LEARNING_RATE * gradients.weights.getWeight(i, j, k).*;
                }
            }
        }
    }
};
