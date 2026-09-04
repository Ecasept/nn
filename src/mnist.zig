const std = @import("std");
const app = @import("mnist/main.zig");

pub fn main(init: std.process.Init) void {
    app.main(init);
}
