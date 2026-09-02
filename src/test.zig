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

    const layers = [_]net.Layer{
        .{ .size = 2, .activation = act.Activations().relu() },
        .{ .size = 3, .activation = act.Activations().relu() },
        .{ .size = 2, .activation = act.Activations().relu() },
    };

    var runner = try nn.NetworkRunner.init(
        allocator,
        io,
        "",
        &layers,
        dl.EMPTY_DATA_LOADER,
    );

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
