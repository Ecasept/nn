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

const LoadDataFn = *const fn (std.mem.Allocator, std.Io, []const u8) anyerror!std.ArrayList(DataPoint);

pub const DataLoader = struct {
    loadData: LoadDataFn,
};

pub fn dataLoaderFrom(loadDataFn: LoadDataFn) DataLoader {
    return .{ .loadData = loadDataFn };
}
pub const EMPTY_DATA_LOADER = dataLoaderFrom(struct {
    fn loadData(allocator: std.mem.Allocator, _: std.Io, _: []const u8) anyerror!std.ArrayList(DataPoint) {
        const list = std.ArrayList(DataPoint).initCapacity(allocator, 0);
        return list;
    }
}.loadData);
