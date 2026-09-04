const std = @import("std");
const neural = @import("../lib/lib.zig");
const net = @import("../lib/network.zig");
const dataLoader = @import("data.zig");
const dataLoaderTypes = @import("../lib/data_loader.zig");
const act = @import("../lib/activation.zig");

const TrainingConfig = struct {
    layers: []const net.Layer,
    trainingDataPath: []const u8,
    testDataPath: []const u8,
    epochs: u64,
    batchSize: usize,
    learningRate: f32,
    randomSeed: u64,
};

const trainingConfig: TrainingConfig = .{
    .layers = &.{
        .{ .size = dataLoader.MNIST_SIZE, .activation = act.Activations().sigmoid() },
        .{ .size = 32, .activation = act.Activations().sigmoid() },
        .{ .size = 10, .activation = act.Activations().sigmoid() },
    },
    .trainingDataPath = "assets/mnist_train.csv",
    .testDataPath = "assets/mnist_test.csv",
    .epochs = 20,
    .batchSize = 32,
    .learningRate = 10,
    .randomSeed = 67,
};

pub fn main(init: std.process.Init) void {
    var threadedIo = std.Io.Threaded.init_single_threaded;
    const io = threadedIo.io();
    train(init.gpa, io) catch |err| {
        std.log.err("err: {}", .{err});
    };
}

pub fn train(allocator: std.mem.Allocator, io: std.Io) !void {
    const REPORT_TEST_ACCURACY = true;
    const dl = dataLoader.getDataLoader();
    var prng = std.Random.DefaultPrng.init(trainingConfig.randomSeed);
    var networkRunner = try neural.NetworkRunner.init(allocator, io, trainingConfig.trainingDataPath, trainingConfig.layers, dl, trainingConfig.batchSize, prng.random());
    defer networkRunner.deinit();
    networkRunner.network.learningRate = trainingConfig.learningRate;

    var testData: ?std.ArrayList(dataLoaderTypes.DataPoint) = if (REPORT_TEST_ACCURACY)
        try dl.loadData(allocator, io, trainingConfig.testDataPath)
    else
        null;
    defer if (testData) |*data| {
        for (data.items) |dataPoint| dataPoint.deinit();
        data.deinit(allocator);
    };

    std.debug.print("Training...\n", .{});
    const start = std.Io.Clock.real.now(io).toMilliseconds();

    var epoch: u64 = 1;
    while (epoch <= trainingConfig.epochs) : (epoch += 1) {
        const cost = networkRunner.trainEpoch();
        std.debug.print("\nEpoch {}\n", .{epoch});
        std.debug.print("Cost: {}\n", .{cost});
        if (testData) |data| {
            std.debug.print("Testing accuracy: {d:.2}%\n", .{networkRunner.accuracy(data.items) * 100});
        }

        const now = std.Io.Clock.real.now(io).toMilliseconds();
        const msPerEpoch = @as(f32, @floatFromInt(now - start)) / @as(f32, @floatFromInt(epoch));
        std.debug.print("ETA: {}s ({}s/epoch)\n", .{ msPerEpoch * @as(f32, @floatFromInt(trainingConfig.epochs - epoch)) / 1000.0, msPerEpoch / 1000.0 });
    }
    if (testData) |data| {
        std.debug.print("Test performance: {d}%\n", .{networkRunner.accuracy(data.items) * 100});
    } else {
        try networkRunner.switchDataset(io, trainingConfig.testDataPath);
        std.debug.print("Test performance: {d}%\n", .{networkRunner.testEpoch() * 100});
    }
}
