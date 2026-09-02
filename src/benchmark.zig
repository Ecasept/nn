const std = @import("std");
const la = @import("lib/la.zig");

const ROUNDS = 10000;
const WARMUP_ROUNDS = 100;

const BenchmarkResult = struct {
    f1: i64,
    f2: i64,
};

fn Benchmark(comptime Context: type) type {
    return struct {
        fn run(context: Context, io: std.Io) BenchmarkResult {
            for (0..WARMUP_ROUNDS) |_| {
                context.runFrog();
                context.runManual();
            }
            var f1Time: i64 = 0;
            var f2Time: i64 = 0;
            for (0..ROUNDS) |_| {
                const startF1 = std.Io.Clock.real.now(io).toMilliseconds();
                context.runFrog();
                f1Time += std.Io.Clock.real.now(io).toMilliseconds() - startF1;
                const startF2 = std.Io.Clock.real.now(io).toMilliseconds();
                context.runManual();
                f2Time += std.Io.Clock.real.now(io).toMilliseconds() - startF2;
            }
            return .{
                .f1 = f1Time,
                .f2 = f2Time,
            };
        }
    };
}

fn testVectorExpression(io: std.Io, allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(67);
    const rand = prng.random();

    const len = rand.intRangeAtMost(usize, 500, 1000);
    const a = try allocator.alloc(f32, len);
    const b = try allocator.alloc(f32, len);
    const c = try allocator.alloc(f32, len);
    const d = try allocator.alloc(f32, len);
    const out = try allocator.alloc(f32, len);
    std.mem.doNotOptimizeAway(out);
    for (0..len) |i| {
        a[i] = rand.float(f32);
        b[i] = rand.float(f32);
        c[i] = rand.float(f32);
        d[i] = rand.float(f32);
    }
    const Context = struct {
        a: []const f32,
        b: []const f32,
        c: []const f32,
        d: []const f32,
        out: []f32,
        fn runFrog(self: @This()) void {
            Impl.vectorExpressionFrog(self.a, self.b, self.c, self.d, self.out);
        }
        fn runManual(self: @This()) void {
            Impl.vectorExpressionManual(self.a, self.b, self.c, self.d, self.out);
        }
    };

    const context: Context = .{
        .a = a,
        .b = b,
        .c = c,
        .d = d,
        .out = out,
    };
    const result = Benchmark(Context).run(context, io);
    const frogTime = result.f1;
    const manualTime = result.f2;

    std.debug.print("Vector expression benchmark:\n", .{});
    std.debug.print("Frog time: {d}ms\n", .{frogTime});
    std.debug.print("Manual time: {d}ms\n", .{manualTime});

    allocator.free(a);
    allocator.free(b);
    allocator.free(c);
    allocator.free(d);
    allocator.free(out);
}
fn testMatrixExpression(io: std.Io, allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(67);
    const rand = prng.random();

    const rows = rand.intRangeAtMost(usize, 50, 100);
    const cols = rand.intRangeAtMost(usize, 50, 100);
    const a = try la.Matrix(f32).init(allocator, cols, rows);
    const b = try la.Matrix(f32).init(allocator, cols, rows);
    const c = try la.Matrix(f32).init(allocator, cols, rows);
    const d = try la.Matrix(f32).init(allocator, cols, rows);
    const out = try la.Matrix(f32).init(allocator, cols, rows);
    std.mem.doNotOptimizeAway(out.data);
    for (0..rows) |row| {
        for (0..cols) |col| {
            a.set(row, col, rand.float(f32));
            b.set(row, col, rand.float(f32));
            c.set(row, col, rand.float(f32));
            d.set(row, col, rand.float(f32));
        }
    }
    const Context = struct {
        a: la.Matrix(f32),
        b: la.Matrix(f32),
        c: la.Matrix(f32),
        d: la.Matrix(f32),
        out: la.Matrix(f32),
        fn runFrog(self: @This()) void {
            Impl.matrixExpressionFrog(self.a, self.b, self.c, self.d, self.out);
        }
        fn runManual(self: @This()) void {
            Impl.matrixExpressionManual(self.a, self.b, self.c, self.d, self.out);
        }
    };
    const context: Context = .{
        .a = a,
        .b = b,
        .c = c,
        .d = d,
        .out = out,
    };
    const result = Benchmark(Context).run(context, io);
    const frogTime = result.f1;
    const manualTime = result.f2;

    std.debug.print("Matrix expression benchmark:\n", .{});
    std.debug.print("Frog time: {d}ms\n", .{frogTime});
    std.debug.print("Manual time: {d}ms\n", .{manualTime});

    allocator.free(a.data);
    allocator.free(b.data);
    allocator.free(c.data);
    allocator.free(d.data);
    allocator.free(out.data);
}
fn testMatmulExpression(io: std.Io, allocator: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(67);
    const rand = prng.random();

    const m = rand.intRangeAtMost(usize, 50, 100);
    const k = rand.intRangeAtMost(usize, 50, 100);
    const n = rand.intRangeAtMost(usize, 50, 100);
    const left = try la.Matrix(f32).init(allocator, k, m);
    const right = try la.Matrix(f32).init(allocator, n, k);
    const skip_left = try la.Matrix(f32).init(allocator, k, m);
    const skip_right = try la.Matrix(f32).init(allocator, n, k);
    const gate = try la.Matrix(f32).init(allocator, n, m);
    const bias = try la.Matrix(f32).init(allocator, n, m);
    const out = try la.Matrix(f32).init(allocator, n, m);
    std.mem.doNotOptimizeAway(out.data);
    for (0..m) |row| {
        for (0..k) |col| {
            left.set(row, col, rand.float(f32));
            skip_left.set(row, col, rand.float(f32));
        }
    }
    for (0..k) |row| {
        for (0..n) |col| {
            right.set(row, col, rand.float(f32));
            skip_right.set(row, col, rand.float(f32));
        }
    }
    for (0..m) |row| {
        for (0..n) |col| {
            gate.set(row, col, rand.float(f32));
            bias.set(row, col, rand.float(f32));
        }
    }
    const Context = struct {
        left: la.Matrix(f32),
        right: la.Matrix(f32),
        skip_left: la.Matrix(f32),
        skip_right: la.Matrix(f32),
        gate: la.Matrix(f32),
        bias: la.Matrix(f32),
        out: la.Matrix(f32),
        fn runFrog(self: @This()) void {
            Impl.matmulExpressionFrog(self.left, self.right, self.skip_left, self.skip_right, self.gate, self.bias, self.out);
        }
        fn runManual(self: @This()) void {
            Impl.matmulExpressionManual(self.left, self.right, self.skip_left, self.skip_right, self.gate, self.bias, self.out);
        }
    };
    const context: Context = .{
        .left = left,
        .right = right,
        .skip_left = skip_left,
        .skip_right = skip_right,
        .gate = gate,
        .bias = bias,
        .out = out,
    };
    const result = Benchmark(Context).run(context, io);
    const frogTime = result.f1;
    const manualTime = result.f2;

    std.debug.print("Matmul expression benchmark:\n", .{});
    std.debug.print("Frog time: {d}ms\n", .{frogTime});
    std.debug.print("Manual time: {d}ms\n", .{manualTime});

    allocator.free(left.data);
    allocator.free(right.data);
    allocator.free(skip_left.data);
    allocator.free(skip_right.data);
    allocator.free(gate.data);
    allocator.free(bias.data);
    allocator.free(out.data);
}

pub fn main(init: std.process.Init) !void {
    var threadedIo = std.Io.Threaded.init_single_threaded;
    const io = threadedIo.io();

    const allocator = init.gpa;
    try testVectorExpression(io, allocator);
    try testMatrixExpression(io, allocator);
    try testMatmulExpression(io, allocator);
}

const Impl = struct {
    /// Computes:
    ///
    ///     out = ((a + b) * (c - d)) + ((a * c) - (b * d))
    ///
    /// The complete expression is evaluated lazily and written to `out` once.
    pub fn vectorExpressionFrog(
        a: []const f32,
        b: []const f32,
        c: []const f32,
        d: []const f32,
        out: []f32,
    ) void {
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == c.len);
        std.debug.assert(a.len == d.len);
        std.debug.assert(a.len == out.len);

        const expression = la.add(
            la.mul(
                la.add(la.from(a), la.from(b)),
                la.sub(la.from(c), la.from(d)),
            ),
            la.sub(
                la.mul(la.from(a), la.from(c)),
                la.mul(la.from(b), la.from(d)),
            ),
        );
        la.store(expression, out);
    }

    /// Loop implementation of `vectorExpressionFrog`.
    pub fn vectorExpressionManual(
        a: []const f32,
        b: []const f32,
        c: []const f32,
        d: []const f32,
        out: []f32,
    ) void {
        std.debug.assert(a.len == b.len);
        std.debug.assert(a.len == c.len);
        std.debug.assert(a.len == d.len);
        std.debug.assert(a.len == out.len);

        for (0..out.len) |i| {
            out[i] = ((a[i] + b[i]) * (c[i] - d[i])) +
                ((a[i] * c[i]) - (b[i] * d[i]));
        }
    }

    /// Computes the element-wise matrix expression:
    ///
    ///     out = ((a + b) * (c - d)) + ((a * d) - (b * c))
    pub fn matrixExpressionFrog(
        a: la.Matrix(f32),
        b: la.Matrix(f32),
        c: la.Matrix(f32),
        d: la.Matrix(f32),
        out: la.Matrix(f32),
    ) void {
        assertSameShape(a, b);
        assertSameShape(a, c);
        assertSameShape(a, d);
        assertSameShape(a, out);

        const expression = la.addMat(
            la.mulMat(
                la.addMat(a, b),
                la.subMat(c, d),
            ),
            la.subMat(
                la.mulMat(a, d),
                la.mulMat(b, c),
            ),
        );
        la.storeMat(expression, out);
    }

    /// Loop implementation of `matrixExpressionFrog`.
    pub fn matrixExpressionManual(
        a: la.Matrix(f32),
        b: la.Matrix(f32),
        c: la.Matrix(f32),
        d: la.Matrix(f32),
        out: la.Matrix(f32),
    ) void {
        assertSameShape(a, b);
        assertSameShape(a, c);
        assertSameShape(a, d);
        assertSameShape(a, out);

        for (0..out.height()) |row| {
            for (0..out.width()) |column| {
                out.set(
                    row,
                    column,
                    ((a.get(row, column) + b.get(row, column)) *
                        (c.get(row, column) - d.get(row, column))) +
                        ((a.get(row, column) * d.get(row, column)) -
                            (b.get(row, column) * c.get(row, column))),
                );
            }
        }
    }

    /// Computes two matrix products and fuses the remaining element-wise work:
    ///
    ///     out = (((left * right) + (skip_left * skip_right)) * gate) + bias
    ///
    /// `left` and `skip_left` have shape M x K, `right` and `skip_right` have
    /// shape K x N, and `gate`, `bias`, and `out` have shape M x N.
    pub fn matmulExpressionFrog(
        left: la.Matrix(f32),
        right: la.Matrix(f32),
        skip_left: la.Matrix(f32),
        skip_right: la.Matrix(f32),
        gate: la.Matrix(f32),
        bias: la.Matrix(f32),
        out: la.Matrix(f32),
    ) void {
        assertMatmulShapes(left, right, out);
        assertMatmulShapes(skip_left, skip_right, out);
        assertSameShape(out, gate);
        assertSameShape(out, bias);

        const expression = la.addMat(
            la.mulMat(
                la.addMat(
                    la.matmul(left, right),
                    la.matmul(skip_left, skip_right),
                ),
                gate,
            ),
            bias,
        );
        la.storeMat(expression, out);
    }

    /// Loop implementation of `matmulExpressionFrog`.
    pub fn matmulExpressionManual(
        left: la.Matrix(f32),
        right: la.Matrix(f32),
        skip_left: la.Matrix(f32),
        skip_right: la.Matrix(f32),
        gate: la.Matrix(f32),
        bias: la.Matrix(f32),
        out: la.Matrix(f32),
    ) void {
        assertMatmulShapes(left, right, out);
        assertMatmulShapes(skip_left, skip_right, out);
        assertSameShape(out, gate);
        assertSameShape(out, bias);

        for (0..out.height()) |row| {
            for (0..out.width()) |column| {
                var product: f32 = 0;
                var skip_product: f32 = 0;

                for (0..left.width()) |inner| {
                    product += left.get(row, inner) * right.get(inner, column);
                }
                for (0..skip_left.width()) |inner| {
                    skip_product += skip_left.get(row, inner) * skip_right.get(inner, column);
                }

                out.set(
                    row,
                    column,
                    ((product + skip_product) * gate.get(row, column)) +
                        bias.get(row, column),
                );
            }
        }
    }

    fn assertSameShape(a: la.Matrix(f32), b: la.Matrix(f32)) void {
        std.debug.assert(a.width() == b.width());
        std.debug.assert(a.height() == b.height());
    }

    fn assertMatmulShapes(
        left: la.Matrix(f32),
        right: la.Matrix(f32),
        out: la.Matrix(f32),
    ) void {
        std.debug.assert(left.width() == right.height());
        std.debug.assert(left.height() == out.height());
        std.debug.assert(right.width() == out.width());
    }
};
