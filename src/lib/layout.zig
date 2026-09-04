const std = @import("std");
const la = @import("la.zig");
const net = @import("network.zig");

pub fn BatchedNeuronLayout(skipFirst: bool) type {
    const offset = if (skipFirst) 1 else 0;
    return struct {
        data: []la.Matrix(f32),
        allocator: std.mem.Allocator,
        pub fn getNeuron(self: @This(), layerIdx: usize, batchIdx: usize, neuronIdx: usize) *f32 {
            return self.data[layerIdx - offset].getPtr(batchIdx, neuronIdx);
        }
        pub fn getLayer(self: @This(), layerIdx: usize) la.Matrix(f32) {
            return self.data[layerIdx - offset];
        }
        pub fn init(allocator: std.mem.Allocator, layers: []const net.Layer, batchSize: usize) !@This() {
            var layout: @This() = .{ .allocator = allocator, .data = undefined };
            layout.data = try allocator.alloc(la.Matrix(f32), layers.len - offset);
            for (offset..layers.len) |i| {
                layout.data[i - offset] = try la.Matrix(f32).init(allocator, layers[i].size, batchSize);
                @memset(layout.data[i - offset].data, 0);
            }
            return layout;
        }
        pub fn deinit(self: @This()) void {
            for (self.data) |mat| {
                mat.deinit();
            }
            self.allocator.free(self.data);
        }
    };
}
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
        pub fn init(allocator: std.mem.Allocator, layers: []const net.Layer) !NeuronLayout(skipFirst) {
            var layout: @This() = .{ .allocator = allocator, .data = undefined };
            layout.data = try allocator.alloc([]f32, layers.len - offset);
            for (offset..layers.len) |i| {
                const data = try allocator.alloc(f32, layers[i].size);
                @memset(data, 0);
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
pub const WeightsLayout = struct {
    data: []la.Matrix(f32),
    allocator: std.mem.Allocator,

    pub fn getWeight(self: WeightsLayout, layerIdx: usize, neuronIdx: usize, connectedNeuronIdx: usize) *f32 {
        return self.data[layerIdx - 1].getPtr(neuronIdx, connectedNeuronIdx);
    }
    pub fn getWeights(self: WeightsLayout, layerIdx: usize, neuronIdx: usize) []f32 {
        return self.data[layerIdx - 1].getRow(neuronIdx);
    }
    pub fn getLayer(self: WeightsLayout, layerIdx: usize) la.Matrix(f32) {
        return self.data[layerIdx - 1];
    }
    pub fn init(allocator: std.mem.Allocator, layers: []const net.Layer, random: std.Random) !WeightsLayout {
        var layout: @This() = .{ .allocator = allocator, .data = undefined };
        layout.data = try allocator.alloc(la.Matrix(f32), layers.len - 1);
        for (1..layers.len) |i| {
            const prevLayerCount = layers[i - 1].size;
            const thisLayerCount = layers[i].size;
            layout.data[i - 1] = try la.Matrix(f32).init(allocator, prevLayerCount, thisLayerCount);
            const fanIn: f32 = @floatFromInt(prevLayerCount);
            const fanOut: f32 = @floatFromInt(thisLayerCount);
            const standardDeviation = switch (layers[i].activation.weightInitialization) {
                .xavier => @sqrt(2.0 / (fanIn + fanOut)),
                .he => @sqrt(2.0 / fanIn),
            };
            for (layout.data[i - 1].data) |*weight| {
                weight.* = random.floatNorm(f32) * standardDeviation;
            }
        }
        return layout;
    }
    pub fn initZeroed(allocator: std.mem.Allocator, layers: []const net.Layer) !WeightsLayout {
        var layout: @This() = .{ .allocator = allocator, .data = undefined };
        layout.data = try allocator.alloc(la.Matrix(f32), layers.len - 1);
        for (1..layers.len) |i| {
            const prevLayerCount = layers[i - 1].size;
            const thisLayerCount = layers[i].size;
            layout.data[i - 1] = try la.Matrix(f32).init(allocator, prevLayerCount, thisLayerCount);
            @memset(layout.data[i - 1].data, 0);
        }
        return layout;
    }
    pub fn deinit(self: WeightsLayout) void {
        for (self.data) |layer| {
            layer.deinit();
        }
        self.allocator.free(self.data);
    }
};
