const std = @import("std");

// idk if i should call this streams iterators sequences generators etc so ill just call it frog

pub fn SliceFrog(comptime baseType: type) type {
    return struct {
        data: []const baseType,
        pub inline fn len(self: @This()) usize {
            return self.data.len;
        }
        pub inline fn get(self: @This(), i: usize) baseType {
            return self.data[i];
        }
    };
}

pub fn from(value: []const f32) SliceFrog(f32) {
    return .{ .data = value };
}

pub fn BinaryFrog(comptime baseType: type, comptime F1: type, comptime F2: type, comptime operation: fn (baseType, baseType) baseType) type {
    return struct {
        f1: F1,
        f2: F2,
        pub inline fn len(self: @This()) usize {
            return self.f1.len();
        }
        pub inline fn get(self: @This(), i: usize) baseType {
            return operation(self.f1.get(i), self.f2.get(i));
        }
    };
}

pub fn BinaryFrogRuntime(comptime baseType: type, comptime F1: type, comptime F2: type) type {
    return struct {
        f1: F1,
        f2: F2,
        operation: *const fn (baseType, baseType) baseType,
        pub inline fn len(self: @This()) usize {
            return self.f1.len();
        }
        pub inline fn get(self: @This(), i: usize) baseType {
            return self.operation(self.f1.get(i), self.f2.get(i));
        }
    };
}

fn AddOp(comptime T: type) fn (T, T) T {
    return struct {
        fn func(a: T, b: T) T {
            return a + b;
        }
    }.func;
}
fn SubOp(comptime T: type) fn (T, T) T {
    return struct {
        fn func(a: T, b: T) T {
            return a - b;
        }
    }.func;
}
fn MulOp(comptime T: type) fn (T, T) T {
    return struct {
        fn func(a: T, b: T) T {
            return a * b;
        }
    }.func;
}

pub fn add(op1: anytype, op2: anytype) BinaryFrog(f32, @TypeOf(op1), @TypeOf(op2), AddOp(f32)) {
    return .{ .f1 = op1, .f2 = op2 };
}
pub fn sub(op1: anytype, op2: anytype) BinaryFrog(f32, @TypeOf(op1), @TypeOf(op2), SubOp(f32)) {
    return .{ .f1 = op1, .f2 = op2 };
}
pub fn mul(op1: anytype, op2: anytype) BinaryFrog(f32, @TypeOf(op1), @TypeOf(op2), MulOp(f32)) {
    return .{ .f1 = op1, .f2 = op2 };
}

pub fn MapFrog(comptime baseType: type, comptime F: type, comptime operation: fn (baseType) baseType) type {
    return struct {
        f: F,
        pub inline fn len(self: @This()) usize {
            return self.f.len();
        }
        pub inline fn get(self: @This(), i: usize) baseType {
            return operation(self.f.get(i));
        }
    };
}

pub fn MapFrogRuntime(comptime baseType: type, comptime F: type) type {
    return struct {
        f: F,
        operation: *const fn (baseType) baseType,
        pub inline fn len(self: @This()) usize {
            return self.f.len();
        }
        pub inline fn get(self: @This(), i: usize) baseType {
            return self.operation(self.f.get(i));
        }
    };
}

pub fn map(operand: anytype, comptime operation: fn (f32) f32) MapFrog(f32, @TypeOf(operand), operation) {
    return .{ .f = operand };
}

pub fn mapRuntime(operand: anytype, operation: *const fn (f32) f32) MapFrogRuntime(f32, @TypeOf(operand)) {
    return .{ .f = operand, .operation = operation };
}

pub fn BinaryFrog2D(comptime baseType: type, comptime F1: type, comptime F2: type, comptime operation: fn (baseType, baseType) baseType) type {
    return struct {
        f1: F1,
        f2: F2,
        pub inline fn width(self: @This()) usize {
            return self.f1.width();
        }
        pub inline fn height(self: @This()) usize {
            return self.f1.height();
        }
        pub inline fn get(self: @This(), row: usize, column: usize) baseType {
            return operation(self.f1.get(row, column), self.f2.get(row, column));
        }
    };
}

/// Binary element-wise 2D frog, runtime function pointer.
pub fn BinaryFrog2DRuntime(comptime baseType: type, comptime F1: type, comptime F2: type) type {
    return struct {
        f1: F1,
        f2: F2,
        operation: *const fn (baseType, baseType) baseType,
        pub inline fn width(self: @This()) usize {
            return self.f1.width();
        }
        pub inline fn height(self: @This()) usize {
            return self.f1.height();
        }
        pub inline fn get(self: @This(), row: usize, column: usize) baseType {
            return self.operation(self.f1.get(row, column), self.f2.get(row, column));
        }
    };
}

pub fn addMat(op1: anytype, op2: anytype) BinaryFrog2D(f32, @TypeOf(op1), @TypeOf(op2), AddOp(f32)) {
    return .{ .f1 = op1, .f2 = op2 };
}
pub fn subMat(op1: anytype, op2: anytype) BinaryFrog2D(f32, @TypeOf(op1), @TypeOf(op2), SubOp(f32)) {
    return .{ .f1 = op1, .f2 = op2 };
}
pub fn mulMat(op1: anytype, op2: anytype) BinaryFrog2D(f32, @TypeOf(op1), @TypeOf(op2), MulOp(f32)) {
    return .{ .f1 = op1, .f2 = op2 };
}

pub fn MapFrog2D(comptime baseType: type, comptime F: type, comptime operation: fn (baseType) baseType) type {
    return struct {
        f: F,
        pub inline fn width(self: @This()) usize {
            return self.f.width();
        }
        pub inline fn height(self: @This()) usize {
            return self.f.height();
        }
        pub inline fn get(self: @This(), row: usize, column: usize) baseType {
            return operation(self.f.get(row, column));
        }
    };
}

pub fn MapFrog2DRuntime(comptime baseType: type, comptime F: type) type {
    return struct {
        f: F,
        operation: *const fn (baseType) baseType,
        pub inline fn width(self: @This()) usize {
            return self.f.width();
        }
        pub inline fn height(self: @This()) usize {
            return self.f.height();
        }
        pub inline fn get(self: @This(), row: usize, column: usize) baseType {
            return self.operation(self.f.get(row, column));
        }
    };
}

pub fn mapMat(operand: anytype, comptime operation: fn (f32) f32) MapFrog2D(f32, @TypeOf(operand), operation) {
    return .{ .f = operand };
}

pub fn mapMatRuntime(operand: anytype, operation: *const fn (f32) f32) MapFrog2DRuntime(f32, @TypeOf(operand)) {
    return .{ .f = operand, .operation = operation };
}

pub fn ScaleFrog2D(comptime baseType: type, comptime F: type) type {
    return struct {
        f: F,
        scalar: baseType,

        pub inline fn width(self: @This()) usize {
            return self.f.width();
        }
        pub inline fn height(self: @This()) usize {
            return self.f.height();
        }
        pub inline fn get(self: @This(), row: usize, column: usize) baseType {
            return self.f.get(row, column) * self.scalar;
        }
    };
}

pub fn scale(operand: anytype, scalar: f32) ScaleFrog2D(f32, @TypeOf(operand)) {
    return .{ .f = operand, .scalar = scalar };
}

pub fn TransposeFrog(comptime baseType: type, comptime F: type) type {
    return struct {
        f: F,
        pub inline fn width(self: @This()) usize {
            return self.f.height();
        }
        pub inline fn height(self: @This()) usize {
            return self.f.width();
        }
        pub inline fn get(self: @This(), row: usize, column: usize) baseType {
            return self.f.get(column, row);
        }
    };
}

pub fn transpose(matrixFrog: anytype) TransposeFrog(f32, @TypeOf(matrixFrog)) {
    return .{ .f = matrixFrog };
}

pub fn RowFrog(comptime baseType: type, comptime F: type) type {
    return struct {
        f: F,
        row: usize,
        pub inline fn len(self: @This()) usize {
            return self.f.width();
        }
        pub inline fn get(self: @This(), column: usize) baseType {
            return self.f.get(self.row, column);
        }
    };
}

pub fn ColumnFrog(comptime baseType: type, comptime F: type) type {
    return struct {
        f: F,
        column: usize,
        pub inline fn len(self: @This()) usize {
            return self.f.height();
        }
        pub inline fn get(self: @This(), r: usize) baseType {
            return self.f.get(r, self.column);
        }
    };
}

pub fn GeneratorFrog(comptime baseType: type, comptime generator: fn (usize) baseType) type {
    return struct {
        length: usize,
        pub inline fn len(self: @This()) usize {
            return self.length;
        }
        pub inline fn get(_: @This(), r: usize) baseType {
            return generator(r);
        }
    };
}

pub fn BroadcastRowFrog(comptime baseType: type, comptime F: type) type {
    return struct {
        f: F,
        _height: usize,
        pub inline fn width(self: @This()) usize {
            return self.f.len();
        }
        pub inline fn height(self: @This()) usize {
            return self._height;
        }
        pub inline fn get(self: @This(), _: usize, column: usize) baseType {
            return self.f.get(column);
        }
    };
}

pub fn broadcastRows(sliceFrog: anytype, height: usize) BroadcastRowFrog(f32, @TypeOf(sliceFrog)) {
    return .{ .f = sliceFrog, ._height = height };
}

pub fn generate(comptime generator: fn (usize) f32, length: usize) GeneratorFrog(f32, generator) {
    return .{ .length = length };
}

pub fn dot(op1: anytype, op2: anytype) f32 {
    std.debug.assert(op1.len() == op2.len());
    var s: f32 = 0;
    for (0..op1.len()) |i| {
        s += op1.get(i) * op2.get(i);
    }
    return s;
}

pub fn sum(data: anytype) f32 {
    var s: f32 = 0;
    for (0..data.len()) |i| {
        s += data.get(i);
    }
    return s;
}

pub fn store(data: anytype, out: []f32) void {
    std.debug.assert(data.len() == out.len);
    for (0..data.len()) |i| {
        out[i] = data.get(i);
    }
}

pub fn storeMat(data: anytype, out: anytype) void {
    std.debug.assert(out.height() == data.height());
    std.debug.assert(out.width() == data.width());
    for (0..data.height()) |r| {
        for (0..data.width()) |c| {
            out.set(r, c, data.get(r, c));
        }
    }
}

pub fn matCol(matrix: anytype, colIdx: usize) ColumnFrog(f32, @TypeOf(matrix)) {
    return .{ .f = matrix, .column = colIdx };
}

pub fn matRow(matrix: anytype, rowIdx: usize) RowFrog(f32, @TypeOf(matrix)) {
    return .{ .f = matrix, .row = rowIdx };
}

pub fn copy(noalias src: []const f32, noalias dst: []f32) void {
    @memcpy(dst, src);
}

pub fn Matrix(comptime T: type) type {
    return GeneralMatrix([]T, T, true);
}
pub fn ConstMatrix(comptime T: type) type {
    return GeneralMatrix([]const T, T, false);
}
pub fn GeneralMatrix(comptime Slice: type, comptime T: type, comptime mutable: bool) type {
    return struct {
        data: Slice,
        _width: usize,
        _height: usize,
        allocator: std.mem.Allocator,

        pub inline fn init(allocator: std.mem.Allocator, w: usize, h: usize) !Matrix(T) {
            var matrix: @This() = .{ .data = undefined, ._width = w, ._height = h, .allocator = allocator };
            matrix.data = try allocator.alloc(T, w * h);
            return matrix;
        }
        pub inline fn deinit(self: @This()) void {
            self.allocator.free(self.data);
        }
        pub inline fn width(self: @This()) usize {
            return self._width;
        }
        pub inline fn height(self: @This()) usize {
            return self._height;
        }
        inline fn index(self: @This(), r: usize, c: usize) usize {
            return r * self._width + c;
        }
        pub inline fn getRow(self: @This(), r: usize) Slice {
            return self.data[self.index(r, 0)..self.index(r + 1, 0)];
        }
        pub inline fn getColumnFrog(self: @This(), c: usize) ColumnFrog(f32, @This()) {
            return .{ .f = self, .column = c };
        }
        pub inline fn getRowFrog(self: @This(), r: usize) RowFrog(f32, @This()) {
            return .{ .f = self, .row = r };
        }
        pub inline fn get(self: @This(), r: usize, c: usize) T {
            return self.data[self.index(r, c)];
        }
        pub inline fn getPtr(self: @This(), r: usize, c: usize) *T {
            return &self.data[self.index(r, c)];
        }
        pub inline fn set(self: @This(), r: usize, c: usize, value: T) void {
            if (mutable) {
                self.data[self.index(r, c)] = value;
            } else {
                @compileError("Matrix is not mutable");
            }
        }
        pub inline fn as1DFrog(self: @This()) SliceFrog(T) {
            return .{ .data = self.data };
        }
        pub inline fn from(data: Slice, w: usize, h: usize) @This() {
            std.debug.assert(data.len == w * h);
            return .{ .data = data, ._width = w, ._height = h, .allocator = undefined };
        }

        pub inline fn view(self: @This(), viewWidth: usize, viewHeight: usize) @This() {
            std.debug.assert(viewWidth * viewHeight <= self.data.len);
            return .{
                .data = self.data[0 .. viewWidth * viewHeight],
                ._width = viewWidth,
                ._height = viewHeight,
                .allocator = undefined, // non-owning
            };
        }
        pub inline fn constView(self: @This(), viewWidth: usize, viewHeight: usize) ConstMatrix(T) {
            std.debug.assert(viewWidth * viewHeight <= self.data.len);
            return .{
                .data = self.data[0 .. viewWidth * viewHeight],
                ._width = viewWidth,
                ._height = viewHeight,
                .allocator = undefined, // non-owning
            };
        }
    };
}

pub fn MatmulFrog(comptime baseType: type, comptime M1: type, comptime M2: type) type {
    return struct {
        m1: M1,
        m2: M2,
        pub inline fn width(self: @This()) usize {
            return self.m2.width();
        }
        pub inline fn height(self: @This()) usize {
            return self.m1.height();
        }
        pub inline fn get(self: @This(), r: usize, c: usize) baseType {
            return dot(matRow(self.m1, r), matCol(self.m2, c));
        }
    };
}

pub fn matmul(m1: anytype, m2: anytype) MatmulFrog(f32, @TypeOf(m1), @TypeOf(m2)) {
    std.debug.assert(m1.width() == m2.height());
    return .{ .m1 = m1, .m2 = m2 };
}

test "elementwise vector addition" {
    const a: [3]f32 = .{ 1, 2, 3 };
    const b: [3]f32 = .{ 4, 5, 6 };
    var c: [3]f32 = .{ undefined, undefined, undefined };
    store(add(from(&a), from(&b)), &c);
    std.debug.assert(c[0] == 5);
    std.debug.assert(c[1] == 7);
    std.debug.assert(c[2] == 9);
}

test "elementwise vector multiplication" {
    const a: [3]f32 = .{ 1, 2, 3 };
    const b: [3]f32 = .{ 4, 5, 6 };
    var c: [3]f32 = .{ undefined, undefined, undefined };
    store(mul(from(&a), from(&b)), &c);
    std.debug.assert(c[0] == 4);
    std.debug.assert(c[1] == 10);
    std.debug.assert(c[2] == 18);
}

test "elementwise vector subtraction" {
    const a: [3]f32 = .{ 1, 2, 3 };
    const b: [3]f32 = .{ 4, 5, 6 };
    var c: [3]f32 = .{ undefined, undefined, undefined };
    store(sub(from(&a), from(&b)), &c);
    std.debug.assert(c[0] == -3);
    std.debug.assert(c[1] == -3);
    std.debug.assert(c[2] == -3);
}

pub fn TestMat(comptime rows: usize, comptime cols: usize) type {
    return struct {
        data: [rows * cols]f32,
        const Self = @This();

        pub fn init(input: [rows][cols]f32) Self {
            var self: Self = undefined;
            for (input, 0..) |row, i| {
                for (row, 0..) |val, j| {
                    self.data[i * cols + j] = val;
                }
            }
            return self;
        }
        pub fn undef() Self {
            const self: Self = undefined;
            return self;
        }

        pub fn matrix(self: *const Self) ConstMatrix(f32) {
            return ConstMatrix(f32).from(&self.data, cols, rows);
        }
        pub fn matrixMut(self: *Self) Matrix(f32) {
            return Matrix(f32).from(&self.data, cols, rows);
        }
    };
}

const TestOps = struct {
    fn square(value: f32) f32 {
        return value * value;
    }

    fn offset(value: f32) f32 {
        return value + 0.5;
    }

    fn sequence(index: usize) f32 {
        return @floatFromInt(index * 2 + 1);
    }
};

test "map" {
    const input = [_]f32{ -2, 0, 3 };
    var output: [input.len]f32 = undefined;

    const mapped = map(from(&input), TestOps.square);
    try std.testing.expectEqual(input.len, mapped.len());
    store(mapped, &output);
    try std.testing.expectEqualSlices(f32, &.{ 4, 0, 9 }, &output);
}

test "mapRuntime" {
    const input = [_]f32{ -2, 0, 3 };
    var output: [input.len]f32 = undefined;
    const operation: *const fn (f32) f32 = TestOps.offset;

    const mapped = mapRuntime(from(&input), operation);
    try std.testing.expectEqual(input.len, mapped.len());
    store(mapped, &output);
    try std.testing.expectEqualSlices(f32, &.{ -1.5, 0.5, 3.5 }, &output);
}

test "mapMat" {
    const backing = TestMat(2, 3).init(.{ .{ -2, 0, 3 }, .{ 4, -5, 1 } });
    const mapped = mapMat(backing.matrix(), TestOps.square);

    try std.testing.expectEqual(@as(usize, 3), mapped.width());
    try std.testing.expectEqual(@as(usize, 2), mapped.height());
    try std.testing.expectEqual(@as(f32, 4), mapped.get(0, 0));
    try std.testing.expectEqual(@as(f32, 0), mapped.get(0, 1));
    try std.testing.expectEqual(@as(f32, 9), mapped.get(0, 2));
    try std.testing.expectEqual(@as(f32, 25), mapped.get(1, 1));
}

test "mapMatRuntime" {
    const backing = TestMat(2, 2).init(.{ .{ -2, 0 }, .{ 3, 4 } });
    const operation: *const fn (f32) f32 = TestOps.offset;
    const mapped = mapMatRuntime(backing.matrix(), operation);

    try std.testing.expectEqual(@as(usize, 2), mapped.width());
    try std.testing.expectEqual(@as(usize, 2), mapped.height());
    try std.testing.expectEqual(@as(f32, -1.5), mapped.get(0, 0));
    try std.testing.expectEqual(@as(f32, 0.5), mapped.get(0, 1));
    try std.testing.expectEqual(@as(f32, 3.5), mapped.get(1, 0));
    try std.testing.expectEqual(@as(f32, 4.5), mapped.get(1, 1));
}

test "scale" {
    const backing = TestMat(2, 2).init(.{ .{ 1, -2 }, .{ 0.5, 4 } });
    const scaled = scale(backing.matrix(), -2);

    try std.testing.expectEqual(@as(usize, 2), scaled.width());
    try std.testing.expectEqual(@as(usize, 2), scaled.height());
    try std.testing.expectEqual(@as(f32, -2), scaled.get(0, 0));
    try std.testing.expectEqual(@as(f32, 4), scaled.get(0, 1));
    try std.testing.expectEqual(@as(f32, -1), scaled.get(1, 0));
    try std.testing.expectEqual(@as(f32, -8), scaled.get(1, 1));
}

test "transpose" {
    const backing = TestMat(2, 3).init(.{ .{ 1, 2, 3 }, .{ 4, 5, 6 } });
    const transposed = transpose(backing.matrix());

    try std.testing.expectEqual(@as(usize, 2), transposed.width());
    try std.testing.expectEqual(@as(usize, 3), transposed.height());
    try std.testing.expectEqual(@as(f32, 1), transposed.get(0, 0));
    try std.testing.expectEqual(@as(f32, 4), transposed.get(0, 1));
    try std.testing.expectEqual(@as(f32, 3), transposed.get(2, 0));
    try std.testing.expectEqual(@as(f32, 6), transposed.get(2, 1));
}

test "generate" {
    const generated = generate(TestOps.sequence, 4);

    try std.testing.expectEqual(@as(usize, 4), generated.len());
    try std.testing.expectEqual(@as(f32, 1), generated.get(0));
    try std.testing.expectEqual(@as(f32, 3), generated.get(1));
    try std.testing.expectEqual(@as(f32, 7), generated.get(3));
}

test "broadcastRows" {
    const row = [_]f32{ 2, -1, 4 };
    const broadcast = broadcastRows(from(&row), 3);

    try std.testing.expectEqual(@as(usize, 3), broadcast.width());
    try std.testing.expectEqual(@as(usize, 3), broadcast.height());
    for (0..broadcast.height()) |r| {
        for (row, 0..) |expected, c| {
            try std.testing.expectEqual(expected, broadcast.get(r, c));
        }
    }
}

test "dot" {
    const a = [_]f32{ 1, -2, 3 };
    const b = [_]f32{ 4, 5, -1 };

    try std.testing.expectEqual(@as(f32, -9), dot(from(&a), from(&b)));
}

test "sum" {
    const values = [_]f32{ 1.5, -2, 4, -0.5 };

    try std.testing.expectEqual(@as(f32, 3), sum(from(&values)));
}

test "matmul" {
    const left_backing = TestMat(2, 3).init(.{ .{ 1, 2, 3 }, .{ 4, 5, 6 } });
    const right_backing = TestMat(3, 2).init(.{ .{ 7, 8 }, .{ 9, 10 }, .{ 11, 12 } });
    const product = matmul(left_backing.matrix(), right_backing.matrix());

    try std.testing.expectEqual(@as(usize, 2), product.width());
    try std.testing.expectEqual(@as(usize, 2), product.height());
    try std.testing.expectEqual(@as(f32, 58), product.get(0, 0));
    try std.testing.expectEqual(@as(f32, 64), product.get(0, 1));
    try std.testing.expectEqual(@as(f32, 139), product.get(1, 0));
    try std.testing.expectEqual(@as(f32, 154), product.get(1, 1));
}

test "elementwise matrix addition" {
    const a = TestMat(2, 3).init(.{ .{ 1, 2, 3 }, .{ 4, 5, 6 } }).matrix();
    const b = TestMat(2, 3).init(.{ .{ 7, 8, 9 }, .{ 10, 11, 12 } }).matrix();
    var _c = TestMat(2, 3).undef();
    var c = _c.matrixMut();

    storeMat(addMat(a, b), c);
    std.debug.assert(c.get(0, 0) == 8);
    std.debug.assert(c.get(0, 1) == 10);
    std.debug.assert(c.get(0, 2) == 12);
    std.debug.assert(c.get(1, 0) == 14);
    std.debug.assert(c.get(1, 1) == 16);
    std.debug.assert(c.get(1, 2) == 18);
}

test "elementwise matrix multiplication" {
    const a = TestMat(2, 3).init(.{ .{ 1, 2, 3 }, .{ 4, 5, 6 } }).matrix();
    const b = TestMat(2, 3).init(.{ .{ 7, 8, 9 }, .{ 10, 11, 12 } }).matrix();
    var _c = TestMat(2, 3).undef();
    var c = _c.matrixMut();

    storeMat(mulMat(a, b), c);
    std.debug.assert(c.get(0, 0) == 7);
    std.debug.assert(c.get(0, 1) == 16);
    std.debug.assert(c.get(0, 2) == 27);
    std.debug.assert(c.get(1, 0) == 40);
    std.debug.assert(c.get(1, 1) == 55);
    std.debug.assert(c.get(1, 2) == 72);
}

test "elementwise matrix subtraction" {
    const a = TestMat(2, 3).init(.{ .{ 1, 2, 3 }, .{ 4, 5, 6 } }).matrix();
    const b = TestMat(2, 3).init(.{ .{ 7, 8, 9 }, .{ 10, 11, 12 } }).matrix();
    var _c = TestMat(2, 3).undef();
    var c = _c.matrixMut();

    storeMat(subMat(a, b), c);
    std.debug.assert(c.get(0, 0) == -6);
    std.debug.assert(c.get(0, 1) == -6);
    std.debug.assert(c.get(0, 2) == -6);
    std.debug.assert(c.get(1, 0) == -6);
    std.debug.assert(c.get(1, 1) == -6);
    std.debug.assert(c.get(1, 2) == -6);
}
