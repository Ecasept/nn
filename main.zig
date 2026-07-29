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

const Network = struct {
    layerCount: u8,
    biases: [][]f32,
    weights: [][][]f32,
    allocator: std.mem.Allocator,
    data: std.ArrayList([]f32),
    otherData: std.ArrayList([][]f32),

    pub fn init(allocator: std.mem.Allocator, layers: []u8) !Network {
        var network = Network{ .layerCount = 0, .biases = {}, .weights = {} };
        network.layerCount = layers.len;
        network.allocator = allocator;
        for (1..layers.len) |layerIndexTooMuch| {
            const layerIndex = layerIndexTooMuch - 1; // because the input layer isn't counted
            const neuronCount = layers[layerIndexTooMuch];

            // Init biases
            const biases = try allocator.alloc(f32, neuronCount);
            network.data.append(allocator, biases);
            network.biases[layerIndex] = biases;
            randomInit(biases);

            // Init weight slices
            const weightSlices = try allocator.alloc([]f32, neuronCount);
            network.otherData.append(allocator, weightSlices);
            network.weights[layerIndex] = weightSlices;

            // Init weights
            const neuronCountBefore = layers[layerIndexTooMuch - 1];
            for (0..neuronCount) |neuronIndex| {
                const weights = try allocator.alloc(f32, neuronCountBefore);
                network.data.append(allocator, weights);
                weightSlices[neuronIndex] = weights;
                randomInit(weights);
            }
        }
        return network;
    }

    pub fn deinit(self: *Network) void {
        for (self.data) |slice| {
            self.allocator.free(slice);
        }
        for (self.otherData) |slice| {
            self.allocator.free(slice);
        }
        self.data.deinit(self.allocator);
        self.otherData.deinit(self.allocator);
    }
};

const epochs = 10;

pub fn activation(x: f32) f32 {
    return x;
}

pub fn simulateNetwork(network: Network, dataPoint: TD) f32 {
    const inputs = dataPoint.data;
    const hiddenLayer = [hiddenLayerSize]f32{};
    for (0..hiddenLayerSize) |h| {
        var val = 0;
        for (0..inputsCount) |i| {
            val += inputs[i] * network.hiddenLayerWeights[h][i];
        }
        val = activation(val + network.hiddenLayerBiases[h]);
        hiddenLayer[h] = val;
    }
    const output = [outputsCount]f32{};
    for (0..outputsCount) |o| {
        var val = 0;
        for (0..hiddenLayerSize) |h| {
            val += hiddenLayer[h] * network.outputWeights[o][h];
        }
        val = activation(val + network.outputBiases[o]);
        output[o] = val;
    }
    return output[0];
}

pub fn calculateLoss(dataPoint: TD, output: f32) f32 {
    return (dataPoint.label - output) ** 2;
}

pub fn applyBackPropagation(network: *Network) void {}

pub fn getAllocator() std.mem.Allocator {
    return std.heap.DebugAllocator(.{}).allocator();
}

pub fn train() void {
    const trainingData = createTrainingData();
    const allocator = getAllocator();
    const network = Network.init(allocator);

    for (0..epochs) |i| {
        for (trainingData) |dataPoint| {
            const output = simulateNetwork(network, dataPoint);
            const err = calculateLoss(dataPoint, output);
            applyBackpropagation(&network, err);
        }
    }
    network.deinit();
}
