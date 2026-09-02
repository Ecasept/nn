const std = @import("std");
const la = @import("lib/la.zig");

const ROUNDS = 1000000;
const WARMUP_ROUNDS = 100;

pub fn main(init: std.process.Init) !void {
    var threadedIo = std.Io.Threaded.init_single_threaded;
    const io = threadedIo.io();

    const allocator = init.gpa;

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
    for (0..WARMUP_ROUNDS) |_| {
        Impl.vectorExpressionFrog(a, b, c, d, out);
        Impl.vectorExpressionManual(a, b, c, d, out);
    }
    var frogTime: i64 = 0;
    var manualTime: i64 = 0;
    for (0..ROUNDS) |_| {
        const startFrog = std.Io.Clock.real.now(io).toMilliseconds();
        Impl.vectorExpressionFrog(a, b, c, d, out);
        frogTime += std.Io.Clock.real.now(io).toMilliseconds() - startFrog;
        const startManual = std.Io.Clock.real.now(io).toMilliseconds();
        Impl.vectorExpressionManual(a, b, c, d, out);
        manualTime += std.Io.Clock.real.now(io).toMilliseconds() - startManual;
    }
    std.debug.print("Vector expression benchmark:\n", .{});
    std.debug.print("Frog time: {d}ms\n", .{frogTime});
    std.debug.print("Manual time: {d}ms\n", .{manualTime});

    allocator.free(a);
    allocator.free(b);
    allocator.free(c);
    allocator.free(d);
    allocator.free(out);
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
