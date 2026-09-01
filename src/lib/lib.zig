const std = @import("std");
const la = @import("la.zig");
const layout = @import("layout.zig");
const dl = @import("data_loader.zig");

const LEARNING_RATE = 0.1;

pub fn leakyReLU(comptime leakyFactor: f32) fn (f32) f32 {
    return struct {
        fn closure(x: f32) f32 {
            return if (x > 0) x else leakyFactor;
        }
    }.closure;
}
pub fn reluDeriv(x: f32) f32 {
    return if (x > 0) 1 else 0;
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1 + std.math.exp(-x));
}
pub fn sigmoidDeriv(x: f32) f32 {
    return sigmoid(x) * (1 - sigmoid(x));
}
pub fn sigmoidDerivFromActivation(x: f32) f32 {
    return x * (1 - x);
}

pub fn activation(x: f32) f32 {
    return sigmoid(x);
}
pub fn activationDeriv(x: f32) f32 {
    return sigmoidDerivFromActivation(x);
}

pub fn maxIndex(comptime T: type, slice: []T, lowest: T) usize {
    var maxIdx: usize = 0;
    var maxVal: T = lowest;
    for (0..slice.len) |i| {
        if (slice[i] > maxVal) {
            maxIdx = i;
            maxVal = slice[i];
        }
    }
    return maxIdx;
}

pub const NetworkRunner = struct {
    layers: []const usize,
    network: Network,
    activations: layout.NeuronLayout(false),
    gradients: Gradients,
    costDerivatives: []f32,
    trainingData: std.ArrayList(dl.DataPoint),
    allocator: std.mem.Allocator,
    dataLoader: dl.DataLoader,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, filename: []const u8, layers: []const usize, dataLoader: dl.DataLoader) !NetworkRunner {
        const network = try Network.init(allocator, layers);
        return NetworkRunner{ .allocator = allocator, .layers = layers, .network = network, .activations = try network.initActivations(), .gradients = try network.initGradients(), .costDerivatives = try allocator.alloc(f32, layers[layers.len - 1]), .trainingData = try dataLoader.loadData(allocator, io, filename), .dataLoader = dataLoader };
    }
    pub fn deinit(self: *NetworkRunner) void {
        self.allocator.free(self.costDerivatives);
        self.activations.deinit();
        self.gradients.deinit();
        self.network.deinit();
        for (self.trainingData.items) |dataPoint| {
            dataPoint.deinit();
        }
        self.trainingData.deinit(self.allocator);
    }
    pub fn trainEpoch(self: NetworkRunner) f32 {
        var accCost: f32 = 0;
        for (self.trainingData.items) |dataPoint| {
            self.network.computeActivations(dataPoint.data, self.activations);
            const err = self.network.calculateError(self.activations, dataPoint.label);
            accCost += err;
            self.network.calculateGradients(self.activations, self.gradients, dataPoint.label, self.costDerivatives);
            self.network.applyBackPropagation(self.gradients);
        }
        return accCost / @as(f32, @floatFromInt(self.trainingData.items.len));
    }
    pub fn testEpoch(self: NetworkRunner) f32 {
        var correct: u64 = 0;
        for (self.trainingData.items) |dataPoint| {
            self.network.computeActivations(dataPoint.data, self.activations);
            const modelOut = self.activations.getLayer(self.layers.len - 1);
            const modelAnswer = maxIndex(f32, modelOut, 0);
            const correctAnswer = maxIndex(f32, dataPoint.label, 0);
            if (modelAnswer == correctAnswer) {
                correct += 1;
            }
        }
        return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(self.trainingData.items.len));
    }
    pub fn switchDataset(self: *NetworkRunner, io: std.Io, filename: []const u8) !void {
        // Free current dataset
        for (self.trainingData.items) |dataPoint| {
            dataPoint.deinit();
        }
        self.trainingData.deinit(self.allocator);
        // Load new dataset
        self.trainingData = try self.dataLoader.loadData(self.allocator, io, filename);
    }
};
