const std = @import("std");

// idk if i should call this streams iterators sequences generators etc so ill just call it frog

test "frogs are zero-cost abstraction" {
    const al = std.testing.allocator;

    var prng = std.Random.DefaultPrng.init(67);
    const rand = prng.random();
    const len = rand.int(usize) % 50;

    const a = try al.alloc(f32, len);
    const b = try al.alloc(f32, len);
    const c = try al.alloc(f32, len);
    defer al.free(a);
    defer al.free(b);
    defer al.free(c);

    for (0..len) |i| {
        a[i] = rand.float(f32);
        b[i] = rand.float(f32);
        c[i] = rand.float(f32);
    }

    const res = sum(add(from(a), add(from(b), from(c))));
    std.debug.print("{d}\n", .{res});

    // Examples
    // matmul(m1, matmul(m2, m3))
}

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

pub fn matRow(f: anytype, r: usize) RowFrog(f32, @TypeOf(f)) {
    return .{ .f = f, .row = r };
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

pub fn matColumn(f: anytype, c: usize) ColumnFrog(f32, @TypeOf(f)) {
    return .{ .f = f, .column = c };
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

pub fn storeMat(data: anytype, out: [][]f32) void {
    std.debug.assert(out.len == data.height());
    for (0..data.height()) |r| {
        std.debug.assert(out[r].len == data.width());
        for (0..data.width()) |c| {
            out[r][c] = data.get(r, c);
        }
    }
}

pub fn copy(noalias src: []const f32, noalias dst: []f32) void {
    @memcpy(dst, src);
}

pub fn Matrix(comptime T: type) type {
    return struct {
        data: []T,
        _width: usize,
        _height: usize,

        pub inline fn width(self: @This()) usize {
            return self._width;
        }
        pub inline fn height(self: @This()) usize {
            return self._height;
        }
        inline fn index(self: @This(), r: usize, c: usize) usize {
            return r * self._width + c;
        }
        pub inline fn getRow(self: @This(), r: usize) []T {
            return self.data[self.index(r, 0)..self.index(r + 1, 0)];
        }
        pub inline fn get(self: @This(), r: usize, c: usize) T {
            return self.data[self.index(r, c)];
        }
        pub inline fn set(self: @This(), r: usize, c: usize, value: T) void {
            self.data[self.index(r, c)] = value;
        }
        pub inline fn as1DFrog(self: @This()) SliceFrog(T) {
            return .{ .data = self.data };
        }
        pub inline fn from(data: []T, w: usize, h: usize) Matrix(T) {
            std.debug.assert(data.len == w * h);
            return .{ .data = data, ._width = w, ._height = h };
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
            return dot(matRow(self.m1, r), matColumn(self.m2, c));
        }
    };
}

pub fn matmul(m1: anytype, m2: anytype) MatmulFrog(f32, @TypeOf(m1), @TypeOf(m2)) {
    std.debug.assert(m1.width() == m2.height());
    return .{ .m1 = m1, .m2 = m2 };
}