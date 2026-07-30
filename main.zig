const std = @import("std");

pub fn main() void {
    train();
}

fn TrainingData(comptime data: type, comptime label: type) type {
    return struct {
        data: data,
        label: label,
    };
}
const TD = TrainingData([2]f32, f32);

pub fn createTrainingData() TD {
    return .{ .{ .data = .{ 0, 0 }, .label = 0 }, .{ .data = .{ 0, 1 }, .label = 1 }, .{ .data = .{ 1, 0 }, .label = 1 }, .{ .data = .{ 1, 0 }, .label = 0 } };
}

pub fn randomInit(slice: []f32) void {
    for (slice) |*el| {
        el.* = 0.123;
    }
}
const Layout = struct {
    pub fn NeuronLayout(skipFirst: bool) type {
        const offset = if (skipFirst) 1 else 0;
        return struct {
            data: [][]f32,
            allocator: std.mem.Allocator,

            pub fn getNeuron(self: NeuronLayout(skipFirst), layerIdx: usize, neuronIdx: usize) *f32 {
                return &self[layerIdx - offset][neuronIdx];
            }
            pub fn getLayer(self: NeuronLayout(skipFirst), layerIdx: usize) []f32 {
                return self[layerIdx - offset];
            }
            pub fn init(allocator: std.mem.Allocator, layers: []usize) NeuronLayout(skipFirst) {
                const layout = .{};
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
            return &self[layerIdx - 1][neuronIdx][connectedNeuronIdx];
        }
        pub fn getWeights(self: WeightsLayout, layerIdx: usize, neuronIdx: usize) []f32 {
            return self[layerIdx - 1][neuronIdx];
        }
        pub fn getLayer(self: WeightsLayout, layerIdx: usize) [][]f32 {
            return self[layerIdx - 1];
        }
        pub fn init(allocator: std.mem.Allocator, layers: []usize) WeightsLayout {
            const layout = .{};
            layout.data = try allocator.alloc([]f32, layers.len - 1);
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
    pub fn copy(noalias src: []f32, noalias dst: []f32) void {
        @memcpy(src, dst);
    }
    pub fn add(src1: []f32, src2: []f32, dst: []f32) void {
        for (0..src1.len) |i| {
            dst[i] = src1[i] + src2[i];
        }
    }
    pub fn apply(src: []f32, op: fn (f32) f32, dst: []f32) void {
        for (0..src.len) |i| {
            dst[i] = op(src[i]);
        }
    }
    pub fn dotProduct(input1: []f32, input2: []f32) f32 {
        var sum = 0;
        for (0..input1.len) |i| {
            sum += input1[i] * input2[i];
        }
        return sum;
    }
    pub fn matrixColumnVectorMul(columnVec: []f32, matrix: [][]f32, columnOut: []f32) void {
        for (0..matrix.len) |i| {
            columnOut[i] = LA.dotProduct(columnVec, matrix[i]);
        }
    }
    pub fn abs(data: []f32) f32 {
        if (data.len == 1) {
            return data[0];
        }
        var sum = 0;
        for (0..data.len) |i| {
            sum += data[i] ** 2;
        }
        return std.math.sqrt(sum);
    }
};

const Gradients = struct {
    weights: Layout.WeightsLayout,
    biases: Layout.NeuronLayout(true)
};

const Network = struct {
    layerCount: u8,
    layers: []usize,
    biases: Layout.NeuronLayout(true),
    weights: Layout.WeightsLayout,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, layers: []usize) !Network {
        var network = .{};
        network.layerCount = layers.len;
        network.layers = layers;
        network.allocator = allocator;
        network.biases = Layout.NeuronLayout(true).init(allocator, layers);
        network.weights = Layout.WeightsLayout.init(allocator, layers);
        return network;
    }

    pub fn deinit(self: *Network) void {
        self.biases.deinit();
        self.weights.deinit();
    }
    pub fn initActivations(self: Network) Layout.NeuronLayout(false) {
        return Layout.NeuronLayout(false).init(self.allocator, layers);
    }
    pub fn initGradients(self: Network) Gradients {
        return .{
            Layout.WeightsLayout.init(self.allocator, self.layers),
            Layout.NeuronLayout(true).init(self.allocator, self.layers),
        };
    }
    pub fn computeActivations(self: Network, inputs: []f32, activations: Layout.NeuronLayout(false)) Layout.NeuronLayout {
        LA.copy(inputs, activations.getLayer(0));
        for (1..self.layerCount) |layerIdx| {
            LA.matrixColumnVectorMul(activations.getLayer(layerIdx - 1), self.weights.getLayer(layerIdx), activations[layerIdx]);
            LA.add(activations.getLayer(layerIdx), self.biases.getLayer(layerIdx), activations.getLayer(layerIdx));
            LA.apply(activations.getLayer(layerIdx), activation, activations.getLayer(layerIdx));
        }
        return activations;
    }
    pub fn calculateError(self: Network, activations: Layout.NeuronLayout(false)) []f32 {
        return LA.abs(activations.getLayer(self.layerCount - 1)) ** 2;
    }
    pub fn calculateGradients(self: Network, activations: Layout.NeuronLayout(false), gradients: Gradients) void {
        const PASS_DOWN_INDEX = 0;
        for ((self.layerCount-1)..1) |layerIdx| {
            const neuronCount = self.layers[layerIdx];
            for (0..neuronCount) |neuronIdx| {
                const chainRule = gradients.weights.getWeight(layerIdx, neuronIdx, PASS_DOWN_INDEX);
                const connectedNeuronCount = gradients.
                for (0..)
            }
        }
    }
};

const epochs = 10;

pub fn activation(x: f32) f32 {
    return x;
}

pub fn getAllocator() std.mem.Allocator {
    return std.heap.DebugAllocator(.{}).allocator();
}

pub fn train() !void {
    const trainingData = createTrainingData();
    const allocator = getAllocator();
    const network = try Network.init(allocator);

    const activations = network.initActivations();
    const gradients = network.initGradients();

    for (0..epochs) |_| {
        for (trainingData) |dataPoint| {
            network.computeActivations(dataPoint.data, activations);
            const err = network.calculateError(activations);
            network.calculateGradients(activations, gradient);
            network.applyBackpropagation(err, gradient);

        }
    }
    activations.deinit();
    gradients.deinit();
    network.deinit();
}
