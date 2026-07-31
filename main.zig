const std = @import("std");

pub fn main(init: std.process.Init) void {
    train(init.gpa) catch |err| {
        std.log.err("err: {}", .{err});
    };
}

fn TrainingData(comptime data: type, comptime label: type) type {
    return struct {
        data: data,
        label: label,
    };
}
const TD = TrainingData([2]f32, [1]f32);

var prng = std.Random.DefaultPrng.init(67);
const rand = prng.random();

pub fn createTrainingData() [4]TD {
    return .{ .{ .data = .{ 0, 0 }, .label = .{0} }, .{ .data = .{ 0, 1 }, .label = .{1} }, .{ .data = .{ 1, 0 }, .label = .{1} }, .{ .data = .{ 1, 0 }, .label = .{0} } };
}

pub fn randomInit(slice: []f32) void {
    for (slice) |*el| {
        el.* = rand.float(f32);
    }
}
const Layout = struct {
    pub fn NeuronLayout(skipFirst: bool) type {
        const offset = if (skipFirst) 1 else 0;
        return struct {
            data: [][]f32,
            allocator: std.mem.Allocator,

            pub fn getNeuron(self: NeuronLayout(skipFirst), layerIdx: usize, neuronIdx: usize) *f32 {
                return &self.data[layerIdx - offset][neuronIdx];
            }
            pub fn getLayer(self: NeuronLayout(skipFirst), layerIdx: usize) []f32 {
                return self.data[layerIdx - offset];
            }
            pub fn init(allocator: std.mem.Allocator, layers: []const usize) !NeuronLayout(skipFirst) {
                var layout: @This() = .{ .allocator = allocator, .data = undefined };
                layout.data = try allocator.alloc([]f32, layers.len - offset);
                for (offset..layers.len) |i| {
                    const data = try allocator.alloc(f32, layers[i]);
                    randomInit(data);
                    layout.data[i - offset] = data;
                }
                return layout;
            }
            pub fn deinit(self: NeuronLayout(skipFirst)) void {
                for (self.data) |slice| {
                    self.allocator.free(slice);
                }
                self.allocator.free(self.data);
            }
        };
    }
    const WeightsLayout = struct {
        data: [][][]f32,
        allocator: std.mem.Allocator,

        pub fn getWeight(self: WeightsLayout, layerIdx: usize, neuronIdx: usize, connectedNeuronIdx: usize) *f32 {
            return &self.data[layerIdx - 1][neuronIdx][connectedNeuronIdx];
        }
        pub fn getWeights(self: WeightsLayout, layerIdx: usize, neuronIdx: usize) []f32 {
            return self.data[layerIdx - 1][neuronIdx];
        }
        pub fn getLayer(self: WeightsLayout, layerIdx: usize) [][]f32 {
            return self.data[layerIdx - 1];
        }
        pub fn init(allocator: std.mem.Allocator, layers: []const usize) !WeightsLayout {
            var layout: @This() = .{ .allocator = allocator, .data = undefined };
            layout.data = try allocator.alloc([][]f32, layers.len - 1);
            for (1..layers.len) |i| {
                const prevLayerCount = layers[i - 1];
                const thisLayerCount = layers[i];
                layout.data[i - 1] = try allocator.alloc([]f32, thisLayerCount);
                for (0..layers[i]) |neuron| {
                    const data = try allocator.alloc(f32, prevLayerCount);
                    randomInit(data);
                    layout.data[i - 1][neuron] = data;
                }
            }
            return layout;
        }
        pub fn deinit(self: WeightsLayout) void {
            for (self.data) |layer| {
                for (layer) |slice| {
                    self.allocator.free(slice);
                }
                self.allocator.free(layer);
            }
            self.allocator.free(self.data);
        }
    };
};

const LA = struct {
    pub fn copy(noalias src: []const f32, noalias dst: []f32) void {
        @memcpy(dst, src);
    }
    pub fn add(src1: []const f32, src2: []const f32, dst: []f32) void {
        for (0..src1.len) |i| {
            dst[i] = src1[i] + src2[i];
        }
    }
    pub fn sum(src: []const f32) f32 {
        var sum_: f32 = 0;
        for (0..src.len) |i| {
            sum_ += src[i];
        }
        return sum_;
    }
    pub fn apply(src: []const f32, op: fn (f32) f32, dst: []f32) void {
        for (0..src.len) |i| {
            dst[i] = op(src[i]);
        }
    }
    pub fn dotProduct(input1: []const f32, input2: []const f32) f32 {
        var sum_: f32 = 0;
        for (0..input1.len) |i| {
            sum_ += input1[i] * input2[i];
        }
        return sum_;
    }
    pub fn columnVecMulMatrix(columnVec: []const f32, matrix: []const []const f32, columnOut: []f32) void {
        for (0..matrix.len) |i| {
            columnOut[i] = LA.dotProduct(columnVec, matrix[i]);
        }
    }
    pub fn abs(data: []const f32) f32 {
        if (data.len == 1) {
            return data[0];
        }
        var sum_: f32 = 0;
        for (0..data.len) |i| {
            sum_ += std.math.pow(f32, data[i], 2);
        }
        return std.math.sqrt(sum_);
    }
    pub fn matrixMulRowVec(matrix: []const []const f32, rowVector: []const f32, out: []f32) void {
        for (0..matrix[0].len) |column| {
            var dotProduct_: f32 = 0;
            for (0..matrix.len) |row| {
                dotProduct_ += matrix[row][column] * rowVector[row];
            }
            out[column] = dotProduct_;
        }
    }
    pub fn rowVecMulColumnVec(rowVec: []const f32, columnVec: []const f32, out: [][]f32) void {
        for (0..rowVec.len) |column| {
            for (0..columnVec.len) |row| {
                out[row][column] = rowVec[column] * columnVec[row];
            }
        }
    }
};

const Gradients = struct {
    weights: Layout.WeightsLayout,
    biases: Layout.NeuronLayout(true),
    pub fn deinit(self: Gradients) void {
        self.weights.deinit();
        self.biases.deinit();
    }
};

const Network = struct {
    layerCount: usize,
    layers: []const usize,
    biases: Layout.NeuronLayout(true),
    weights: Layout.WeightsLayout,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, layers: []const usize) !Network {
        return Network{ .layerCount = layers.len, .layers = layers, .allocator = allocator, .biases = try Layout.NeuronLayout(true).init(allocator, layers), .weights = try Layout.WeightsLayout.init(allocator, layers) };
    }

    pub fn deinit(self: *const Network) void {
        self.biases.deinit();
        self.weights.deinit();
    }
    pub fn initActivations(self: Network) !Layout.NeuronLayout(false) {
        return try Layout.NeuronLayout(false).init(self.allocator, self.layers);
    }
    pub fn initGradients(self: Network) !Gradients {
        return .{
            .weights = try Layout.WeightsLayout.init(self.allocator, self.layers),
            .biases = try Layout.NeuronLayout(true).init(self.allocator, self.layers),
        };
    }
    pub fn computeActivations(self: Network, inputs: []const f32, activations: Layout.NeuronLayout(false)) void {
        LA.copy(inputs, activations.getLayer(0));
        for (1..self.layerCount) |layerIdx| {
            LA.columnVecMulMatrix(activations.getLayer(layerIdx - 1), self.weights.getLayer(layerIdx), activations.getLayer(layerIdx));
            LA.add(activations.getLayer(layerIdx), self.biases.getLayer(layerIdx), activations.getLayer(layerIdx));
            LA.apply(activations.getLayer(layerIdx), activation, activations.getLayer(layerIdx));
        }
    }
    pub fn calculateError(self: Network, activations: Layout.NeuronLayout(false), correctOutput: []const f32) f32 {
        var err: f32 = 0;
        const modelOutput = activations.getLayer(self.layerCount - 1);
        if (modelOutput.len == 1) {
            err = modelOutput[0] - correctOutput[0];
        } else {
            for (0..modelOutput.len) |i| {
                const val = modelOutput[i] - correctOutput[i];
                err += std.math.pow(f32, val, 2);
            }
            err = std.math.sqrt(err);
        }
        return std.math.pow(f32, err, 2);
    }
    pub fn calculateErrorDerivative(self: Network, activations: Layout.NeuronLayout(false), correctOutput: []const f32) f32 {
        var err: f32 = 0;
        const modelOutput = activations.getLayer(self.layerCount - 1);
        if (modelOutput.len == 1) {
            err = modelOutput[0] - correctOutput[0];
        } else {
            for (0..modelOutput.len) |i| {
                const val = modelOutput[i] - correctOutput[i];
                err += std.math.pow(f32, val, 2);
            }
            err = std.math.sqrt(err);
        }
        return 2 * err;
    }
    pub fn calculateGradients(self: Network, activations: Layout.NeuronLayout(false), gradients: Gradients, costDerivative: f32) void {
        var layerIdx = self.layerCount;
        while (layerIdx > 1) {
            layerIdx -= 1;
            const biasGradients = gradients.biases.getLayer(layerIdx);
            if (layerIdx != self.layerCount - 1) {
                const nextLayerBiasDerivatives = gradients.biases.getLayer(layerIdx + 1);
                // We want to get dC/db
                // if we change the bias by db, the activation changes by sigma'(activation)
                // so da/db = sigma'(activation)
                // and dC/db = dC/da * da/db
                // and dC/da is just w *

                // weight derivative: a * o'(a) * w * o'(a) * w....
                // bias derivative:       o'(a) * w * o'(a) * w....
                // so to get the bias derivative:
                // o'(a) * (\sum of previous neurons: bias derivative * w)
                // so now for all neurons:
                const nextLayerWeights = self.weights.getLayer(layerIdx + 1);

                // Set bias gradient
                LA.matrixMulRowVec(nextLayerWeights, nextLayerBiasDerivatives, biasGradients);
                for (0..biasGradients.len) |i| {
                    biasGradients[i] = activationDeriv(activations.getNeuron(layerIdx, i).*) * biasGradients[i];
                }
            } else {
                // For the input layer, all the other terms don't exist yet
                for (0..biasGradients.len) |i| {
                    biasGradients[i] = activationDeriv(activations.getNeuron(layerIdx, i).*) * costDerivative;
                }
            }

            // And now for the weighs we just need to multiply by the activations of the previous layer
            LA.rowVecMulColumnVec(activations.getLayer(layerIdx - 1), biasGradients, gradients.weights.getLayer(layerIdx));
        }
    }
    pub fn applyBackPropagation(self: Network, gradients: Gradients) void {
        const eta = 0.001;
        for (1..self.layerCount) |i| {
            for (0..self.layers[i]) |j| {
                self.biases.getNeuron(i, j).* -= eta * gradients.biases.getNeuron(i, j).*;
            }
        }
        for (1..self.layerCount) |i| {
            for (0..self.layers[i]) |j| {
                for (0..self.layers[i - 1]) |k| {
                    self.weights.getWeight(i, j, k).* -= eta * gradients.weights.getWeight(i, j, k).*;
                }
            }
        }
    }
};

const epochs = 10000;

pub fn activation(x: f32) f32 {
    return if (x > 0) x else 0.1;
}
pub fn activationDeriv(x: f32) f32 {
    return if (x > 0) 1 else 0;
}

pub fn train(allocator: std.mem.Allocator) !void {
    const trainingData = createTrainingData();
    const layers = [_]usize{ 2, 3, 1 };
    const network = try Network.init(allocator, &layers);

    const activations = try network.initActivations();
    const gradients = try network.initGradients();

    for (0..epochs) |_| {
        for (trainingData) |dataPoint| {
            network.computeActivations(&dataPoint.data, activations);
            const err = network.calculateError(activations, &dataPoint.label);
            const costDeriv = network.calculateErrorDerivative(activations, &dataPoint.label);
            network.calculateGradients(activations, gradients, costDeriv);
            network.applyBackPropagation(gradients);
            std.debug.print("Error: {}\n", .{err});
            std.debug.print("Input: {any}, Output: {any}, Net: {any}\n", .{ dataPoint.data, dataPoint.label, activations.getLayer(network.layerCount - 1) });
        }
    }
    activations.deinit();
    gradients.deinit();
    network.deinit();
}
