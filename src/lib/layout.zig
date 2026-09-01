const std = @import("std");

var prng = std.Random.DefaultPrng.init(67);
const rand = prng.random();

pub fn randomInit(slice: []f32) void {
    for (slice) |*el| {
        el.* = rand.float(f32) - 0.5;
    }
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
pub const WeightsLayout = struct {
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
