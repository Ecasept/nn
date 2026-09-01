const std = @import("std");

pub const DataPoint = struct {
    data: []f32,
    label: []f32,
    allocator: std.mem.Allocator,
    pub fn deinit(self: DataPoint) void {
        self.allocator.free(self.data);
        self.allocator.free(self.label);
    }
};

pub const DataLoader = struct {
    loadData: fn (std.mem.Allocator, std.Io, []const u8) []DataPoint,
};
