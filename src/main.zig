const std = @import("std");
const neural = @import("lib/lib.zig");
const dataLoader = @import("data.zig");

const EPOCHS = 10;
const PROGRESS_COUNT = EPOCHS;

pub fn main(init: std.process.Init) void {
    var threadedIo = std.Io.Threaded.init_single_threaded;
    const io = threadedIo.io();

    train(init.gpa, io) catch |err| {
        std.log.err("err: {}", .{err});
    };
}

pub fn train(allocator: std.mem.Allocator, io: std.Io) !void {
    const layers = [_]usize{ dataLoader.MNIST_SIZE, 32, 10 };
    const dl = dataLoader.getDataLoader();
    var networkRunner = try neural.NetworkRunner.init(allocator, io, "mnist_train.csv", &layers, dl);
    defer networkRunner.deinit();

    std.debug.print("Training...\n", .{});
    const start = std.Io.Clock.real.now(io).toMilliseconds();

    var epoch: u64 = 1;
    while (epoch <= EPOCHS) : (epoch += 1) {
        const cost = networkRunner.trainEpoch();
        if (epoch % @divExact(EPOCHS, PROGRESS_COUNT) == 0) {
            std.debug.print("\n", .{});
            std.debug.print("Epoch {}\n", .{epoch});
            std.debug.print("Cost: {}\n", .{cost});
            // std.debug.print("Input: {any}, Output: {any}, Net: {any}\n", .{ dataPoint.data, dataPoint.label, activations.getLayer(network.layerCount - 1) });
            // std.debug.print("Weights: {any}, Biases: {any}\n", .{ network.weights.data, network.biases.data });
            // std.debug.print("Gradients: {any}, and {any}\n", .{ gradients.weights.data, gradients.biases.data });

            const now = std.Io.Clock.real.now(io).toMilliseconds();
            const msPerEpoch = @as(f32, @floatFromInt(now - start)) / @as(f32, @floatFromInt(epoch));
            std.debug.print("ETA: {}s ({}s/epoch)\n", .{ msPerEpoch * @as(f32, @floatFromInt(EPOCHS - epoch)) / 1000.0, msPerEpoch / 1000.0 });
        }
    }
    try networkRunner.switchDataset(io, "mnist_test.csv");
    std.debug.print("Test performance: {d}%\n", .{networkRunner.testEpoch() * 100});
    networkRunner.deinit();
}
