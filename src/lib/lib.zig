const std = @import("std");
const la = @import("la.zig");
const layout = @import("layout.zig");
const dl = @import("data_loader.zig");
const network = @import("network.zig");

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
    layers: []const network.Layer,
    network: network.Network,
    activations: layout.BatchedNeuronLayout(false),
    gradients: network.Gradients,
    batchInputs: la.Matrix(f32),
    batchLabels: la.Matrix(f32),
    costDerivatives: la.Matrix(f32),
    trainingData: std.ArrayList(dl.DataPoint),
    allocator: std.mem.Allocator,
    dataLoader: dl.DataLoader,
    batchSize: usize,
    random: std.Random,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, filename: []const u8, layers: []const network.Layer, dataLoader: dl.DataLoader, batchSize: usize, random: std.Random) !NetworkRunner {
        const n = try network.Network.init(allocator, layers, batchSize, random);
        return NetworkRunner{
            .allocator = allocator,
            .layers = layers,
            .network = n,
            .activations = try n.initActivations(batchSize),
            .gradients = try n.initGradients(),
            .batchInputs = try la.Matrix(f32).init(allocator, layers[0].size, batchSize),
            .batchLabels = try la.Matrix(f32).init(allocator, layers[layers.len - 1].size, batchSize),
            .costDerivatives = try n.initCostDerivatives(batchSize),
            .trainingData = try dataLoader.loadData(allocator, io, filename),
            .dataLoader = dataLoader,
            .batchSize = batchSize,
            .random = random,
        };
    }
    pub fn deinit(self: *NetworkRunner) void {
        self.costDerivatives.deinit();
        self.batchLabels.deinit();
        self.batchInputs.deinit();
        self.activations.deinit();
        self.gradients.deinit();
        self.network.deinit();
        for (self.trainingData.items) |dataPoint| {
            dataPoint.deinit();
        }
        self.trainingData.deinit(self.allocator);
    }
    fn activationView(self: NetworkRunner, batchSize: usize) layout.BatchedNeuronLayout(false) {
        const matrices = self.allocator.alloc(la.Matrix(f32), self.layers.len) catch @panic("out of memory");
        for (matrices, 0..) |*matrix, layerIdx| {
            matrix.* = self.activations.getLayer(layerIdx).view(self.layers[layerIdx].size, batchSize);
        }
        return .{ .data = matrices, .allocator = self.allocator };
    }

    fn releaseActivationView(self: NetworkRunner, activations: layout.BatchedNeuronLayout(false)) void {
        self.allocator.free(activations.data);
    }

    fn loadBatch(self: NetworkRunner, dataPoints: []const dl.DataPoint) void {
        for (dataPoints, 0..) |dataPoint, batchIdx| {
            @memcpy(self.batchInputs.getRow(batchIdx), dataPoint.data);
            @memcpy(self.batchLabels.getRow(batchIdx), dataPoint.label);
        }
    }

    pub fn trainEpoch(self: NetworkRunner) f32 {
        // fisher-yates
        self.random.shuffle(dl.DataPoint, self.trainingData.items);

        var accCost: f32 = 0;
        var batch: usize = 0;
        while (batch < self.trainingData.items.len) : (batch += self.batchSize) {
            const batchEnd = @min(batch + self.batchSize, self.trainingData.items.len);
            const dataPoints = self.trainingData.items[batch..batchEnd];
            const currentBatchSize = dataPoints.len;
            const isLastBatch = currentBatchSize < self.batchSize;
            self.loadBatch(dataPoints);

            const inputs = self.batchInputs.constView(self.layers[0].size, currentBatchSize);
            const labels = self.batchLabels.constView(self.layers[self.layers.len - 1].size, currentBatchSize);
            const costDerivatives = self.costDerivatives.view(self.layers[self.layers.len - 1].size, currentBatchSize);
            var activations: layout.BatchedNeuronLayout(false) = undefined;
            if (isLastBatch) {
                // If the last batch is smaller than the batch size, we need to create a new activation view with the correct size
                const newActivations = self.activationView(currentBatchSize);
                activations = newActivations;
            } else {
                activations = self.activations;
            }

            self.network.computeActivations(inputs, activations);
            const err = self.network.calculateMSE(activations, labels);
            accCost += err * @as(f32, @floatFromInt(currentBatchSize));
            self.network.calculateGradients(activations, self.gradients, labels, costDerivatives, currentBatchSize);
            self.network.applyBackPropagation(self.gradients);

            if (isLastBatch) {
                self.releaseActivationView(activations);
            }
        }
        return accCost / @as(f32, @floatFromInt(self.trainingData.items.len));
    }
    pub fn testEpoch(self: NetworkRunner) f32 {
        return self.accuracy(self.trainingData.items);
    }

    pub fn accuracy(self: NetworkRunner, data: []const dl.DataPoint) f32 {
        var correct: u64 = 0;
        var batch: usize = 0;
        while (batch < data.len) : (batch += self.batchSize) {
            const batchEnd = @min(batch + self.batchSize, data.len);
            const dataPoints = data[batch..batchEnd];
            const isLastBatch = dataPoints.len < self.batchSize;
            self.loadBatch(dataPoints);

            const inputs = self.batchInputs.constView(self.layers[0].size, dataPoints.len);
            var activations: layout.BatchedNeuronLayout(false) = undefined;
            if (isLastBatch) {
                // If the last batch is smaller than the batch size, we need to create a new activation view with the correct size
                const newActivations = self.activationView(dataPoints.len);
                activations = newActivations;
            } else {
                activations = self.activations;
            }
            self.network.computeActivations(inputs, activations);

            const modelOut = activations.getLayer(self.layers.len - 1);
            for (dataPoints, 0..) |dataPoint, batchIdx| {
                const modelAnswer = maxIndex(f32, modelOut.getRow(batchIdx), 0);
                const correctAnswer = maxIndex(f32, dataPoint.label, 0);
                if (modelAnswer == correctAnswer) correct += 1;
            }

            if (isLastBatch) {
                self.releaseActivationView(activations);
            }
        }
        return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(data.len));
    }

    /// Must call `self.releaseActivationView(activations)` after using the returned activations.
    /// Computes the activations for a single input instance (instead of a batch) and returns the activations. The returned activations are a view of the internal activations, so they should not be modified.
    fn computeActivationsSingleInstance(self: NetworkRunner, input: []const f32) layout.BatchedNeuronLayout(false) {
        @memcpy(self.batchInputs.getRow(0), input);
        const inputs = self.batchInputs.constView(self.layers[0].size, 1);
        const activations = self.activationView(1);
        self.network.computeActivations(inputs, activations);
        return activations;
    }
    /// Copies the model output for a single input instance into the provided output slice.
    pub fn getModelOutput(self: NetworkRunner, input: []const f32, output: []f32) void {
        const activations = self.computeActivationsSingleInstance(input);
        const modelOutput = activations.getLayer(self.layers.len - 1).getRow(0);
        @memcpy(output, modelOutput);
        self.releaseActivationView(activations);
    }
    /// Predicts the class for a single input instance based on the maximum output neuron value. Returns the index of the predicted class.
    pub fn predict(self: NetworkRunner, input: []const f32) usize {
        const activations = self.computeActivationsSingleInstance(input);
        const modelOutput = activations.getLayer(self.layers.len - 1).getRow(0);
        const prediction = maxIndex(f32, modelOutput, 0);
        self.releaseActivationView(activations);
        return prediction;
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
