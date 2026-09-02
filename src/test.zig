const std = @import("std");

const la = @import("lib/la.zig");

test {
    _ = la;
}

test "addition" {
    try std.testing.expectEqual(@as(i32, 5), 2 + 3);
}
