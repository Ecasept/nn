const std = @import("std");

pub const MNIST_SIZE = 784;

pub const DataPoint = struct {
    data: []f32,
    label: []f32,
    allocator: std.mem.Allocator,
    pub fn deinit(self: DataPoint) void {
        self.allocator.free(self.data);
        self.allocator.free(self.label);
    }
};

const CSVError = error{
    Malformed,
};

pub fn loadData(allocator: std.mem.Allocator, io: std.Io, filename: []const u8) !std.ArrayList(DataPoint) {
    std.debug.print("Loading data...\n", .{});
    const buffer = try std.Io.Dir.cwd().readFileAlloc(io, filename, allocator, std.Io.Limit.unlimited);
    defer allocator.free(buffer);

    var dataList = try std.ArrayList(DataPoint).initCapacity(allocator, 100);

    var rowIter = std.mem.splitSequence(u8, buffer, "\n");
    while (rowIter.next()) |row| {
        if (row.len == 0) {
            continue;
        }
        var colIter = std.mem.splitSequence(u8, row, ",");
        const labelStr = colIter.next() orelse continue;
        const labelScalar = try std.fmt.parseInt(u8, labelStr, 10);
        const label = try allocator.alloc(f32, 10);
        // one hot encoding
        for (0..10) |i| {
            label[i] = if (i == labelScalar) 1 else 0;
        }

        const data = try allocator.alloc(f32, MNIST_SIZE);
        for (0..MNIST_SIZE) |i| {
            const str = colIter.next() orelse return CSVError.Malformed;
            const num = try std.fmt.parseInt(u8, str, 10);
            data[i] = @as(f32, @floatFromInt(num)) / 255.0;
        }
        const dataPoint = DataPoint{
            .data = data,
            .label = label,
            .allocator = allocator,
        };
        try dataList.append(allocator, dataPoint);
    }
    return dataList;
}
