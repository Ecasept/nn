const std = @import("std");
const nn = @import("lib/lib.zig");
const la = @import("lib/la.zig");
const dl = @import("lib/data_loader.zig");
const act = @import("lib/activation.zig");
const net = @import("lib/network.zig");

test {
    _ = la;
}

test "explicit forward pass" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(67);

    const layers = [_]net.Layer{
        .{ .size = 2, .activation = act.Activations().relu() },
        .{ .size = 3, .activation = act.Activations().relu() },
        .{ .size = 2, .activation = act.Activations().relu() },
    };

    var runner = try nn.NetworkRunner.init(allocator, io, "", &layers, dl.EMPTY_DATA_LOADER, 1, prng.random());

    // Weights going to first neuron of second layer
    runner.network.weights.getWeight(1, 0, 0).* = 0.5;
    runner.network.weights.getWeight(1, 0, 1).* = 1.0;

    // Weights going to second neuron of second layer
    runner.network.weights.getWeight(1, 1, 0).* = -0.5;
    runner.network.weights.getWeight(1, 1, 1).* = 0.5;

    // Weights going to third neuron of second layer
    runner.network.weights.getWeight(1, 2, 0).* = 1.0;
    runner.network.weights.getWeight(1, 2, 1).* = -1.0;

    // Weights going to first neuron of third layer
    runner.network.weights.getWeight(2, 0, 0).* = 1.0;
    runner.network.weights.getWeight(2, 0, 1).* = 0.5;
    runner.network.weights.getWeight(2, 0, 2).* = -0.5;

    // Weights going to second neuron of third layer
    runner.network.weights.getWeight(2, 1, 0).* = -0.5;
    runner.network.weights.getWeight(2, 1, 1).* = 1.0;
    runner.network.weights.getWeight(2, 1, 2).* = 0.5;

    // Biases of the neurons in the second layer
    runner.network.biases.getNeuron(1, 0).* = 0.5;
    runner.network.biases.getNeuron(1, 1).* = 1.0;
    runner.network.biases.getNeuron(1, 2).* = 0.0;

    // Biases of the neurons in the third layer
    runner.network.biases.getNeuron(2, 0).* = 0.25;
    runner.network.biases.getNeuron(2, 1).* = 0.5;

    const input = [_]f32{ 1.0, 2.0 };
    var output = [_]f32{ 0.0, 0.0 };

    runner.getModelOutput(&input, &output);

    try std.testing.expectEqual(@as(f32, 4.0), output[0]);
    try std.testing.expectEqual(@as(f32, 0.5), output[1]);

    runner.deinit();
}

test "numerical gradient check" {
    const inputs = [_][2]f32{
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
        .{ 5.0, 6.0 },
    };
    const labels = [_][10]f32{
        .{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 },
        .{ 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 },
        .{ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 },
    };

    const inputsMatrix = la.TestMat(3, 2).init(inputs).matrix();
    const labelsMatrix = la.TestMat(3, 10).init(labels).matrix();

    const allocator = std.testing.allocator;

    const layers = [_]net.Layer{
        .{ .size = 2, .activation = act.Activations().relu() },
        .{ .size = 20, .activation = act.Activations().sigmoid() },
        .{ .size = 30, .activation = act.Activations().relu() },
        .{ .size = 20, .activation = act.Activations().sigmoid() },
        .{ .size = 10, .activation = act.Activations().relu() },
    };
    const BATCH_SIZE = inputs.len;
    const EPSILON = 0.001;
    const ABSOLUTE_ERROR_THRESHOLD = 0.0001;
    const RELATIVE_ERROR_THRESHOLD = 0.01;
    try numericalGradientCheck(allocator, &layers, inputsMatrix, labelsMatrix, BATCH_SIZE, EPSILON, ABSOLUTE_ERROR_THRESHOLD, RELATIVE_ERROR_THRESHOLD);
}

test "learns XOR" {
    const inputsData = [_][2]f32{
        .{ 0, 0 },
        .{ 0, 1 },
        .{ 1, 0 },
        .{ 1, 1 },
    };
    const labelsData = [_][1]f32{
        .{0},
        .{1},
        .{1},
        .{0},
    };
    const inputs = la.TestMat(4, 2).init(inputsData).matrix();
    const labels = la.TestMat(4, 1).init(labelsData).matrix();
    const layers = [_]net.Layer{
        .{ .size = 2, .activation = act.Activations().sigmoid() },
        .{ .size = 4, .activation = act.Activations().sigmoid() },
        .{ .size = 1, .activation = act.Activations().sigmoid() },
    };

    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(67);
    var network = try net.Network.init(allocator, &layers, inputsData.len, prng.random());
    const activations = try network.initActivations(inputsData.len);
    const gradients = try network.initGradients();
    const costDerivatives = try network.initCostDerivatives(inputsData.len);
    defer network.deinit();
    defer activations.deinit();
    defer gradients.deinit();
    defer costDerivatives.deinit();

    const hiddenWeights = [_][2]f32{
        .{ 0.5, -0.4 },
        .{ -0.3, 0.2 },
        .{ 0.1, 0.6 },
        .{ -0.7, -0.2 },
    };
    const hiddenBiases = [_]f32{ 0.1, -0.2, 0.05, 0.3 };
    const outputWeights = [_]f32{ 0.4, -0.5, 0.3, 0.2 };
    for (hiddenWeights, 0..) |weights, neuronIdx| {
        for (weights, 0..) |weight, inputIdx| {
            network.weights.getWeight(1, neuronIdx, inputIdx).* = weight;
        }
        network.biases.getNeuron(1, neuronIdx).* = hiddenBiases[neuronIdx];
        network.weights.getWeight(2, 0, neuronIdx).* = outputWeights[neuronIdx];
    }
    network.biases.getNeuron(2, 0).* = -0.1;
    network.learningRate = 1.0;

    network.computeActivations(inputs, activations);
    const initialLoss = network.calculateMSE(activations, labels);
    for (0..2_000) |_| {
        network.calculateGradients(activations, gradients, labels, costDerivatives, inputsData.len);
        network.applyBackPropagation(gradients);
        network.computeActivations(inputs, activations);
    }
    const finalLoss = network.calculateMSE(activations, labels);

    try std.testing.expect(finalLoss < initialLoss * 0.1);
    const output = activations.getLayer(layers.len - 1);
    for (labelsData, 0..) |expected, sampleIdx| {
        const prediction: f32 = if (output.get(sampleIdx, 0) >= 0.5) 1 else 0;
        try std.testing.expectEqual(expected[0], prediction);
    }
}

fn numericalGradientCheck(allocator: std.mem.Allocator, layers: []const net.Layer, inputs: la.ConstMatrix(f32), labels: la.ConstMatrix(f32), BATCH_SIZE: usize, EPSILON: f32, ABSOLUTE_ERROR_THRESHOLD: f32, RELATIVE_ERROR_THRESHOLD: f32) !void {
    var prng = std.Random.DefaultPrng.init(67);
    var network = try net.Network.init(allocator, layers, BATCH_SIZE, prng.random());
    const activations = try network.initActivations(BATCH_SIZE);
    const normalGradients = try network.initGradients();
    const costDerivatives = try network.initCostDerivatives(BATCH_SIZE);
    defer network.deinit();
    defer activations.deinit();
    defer normalGradients.deinit();
    defer costDerivatives.deinit();

    network.computeActivations(inputs, activations);
    network.calculateGradients(activations, normalGradients, labels, costDerivatives, BATCH_SIZE);

    for (1..network.layerCount) |layerIdx| {
        for (0..network.layers[layerIdx].size) |neuronIdx| {
            for (0..network.layers[layerIdx - 1].size) |connectedNeuronIdx| {
                const weightPtr = network.weights.getWeight(layerIdx, neuronIdx, connectedNeuronIdx);
                const originalWeight = weightPtr.*;

                weightPtr.* = originalWeight - EPSILON;
                network.computeActivations(inputs, activations);
                const minusCost = network.calculateMSE(activations, labels);

                weightPtr.* = originalWeight + EPSILON;
                network.computeActivations(inputs, activations);
                const plusCost = network.calculateMSE(activations, labels);

                const backpropGradient = normalGradients.weights.getWeight(layerIdx, neuronIdx, connectedNeuronIdx).*;
                const numericalGradient = (plusCost - minusCost) / (2 * EPSILON);
                const diff = @abs(numericalGradient - backpropGradient);
                const magnitude = @abs(numericalGradient) + @abs(backpropGradient);
                const relativeError = if (magnitude == 0.0) 0.0 else diff / magnitude;
                try std.testing.expect(diff < ABSOLUTE_ERROR_THRESHOLD or relativeError < RELATIVE_ERROR_THRESHOLD);

                weightPtr.* = originalWeight;
            }
        }
    }
    for (1..network.layerCount) |layerIdx| {
        for (0..network.layers[layerIdx].size) |neuronIdx| {
            const biasPtr = network.biases.getNeuron(layerIdx, neuronIdx);
            const originalBias = biasPtr.*;

            biasPtr.* = originalBias - EPSILON;
            network.computeActivations(inputs, activations);
            const minusCost = network.calculateMSE(activations, labels);

            biasPtr.* = originalBias + EPSILON;
            network.computeActivations(inputs, activations);
            const plusCost = network.calculateMSE(activations, labels);

            const backpropGradient = normalGradients.biases.getNeuron(layerIdx, neuronIdx).*;
            const numericalGradient = (plusCost - minusCost) / (2 * EPSILON);
            const diff = @abs(numericalGradient - backpropGradient);
            const magnitude = @abs(numericalGradient) + @abs(backpropGradient);
            const relativeError = if (magnitude == 0.0) 0.0 else diff / magnitude;

            try std.testing.expect(diff < ABSOLUTE_ERROR_THRESHOLD or relativeError < RELATIVE_ERROR_THRESHOLD);

            biasPtr.* = originalBias;
        }
    }
}
