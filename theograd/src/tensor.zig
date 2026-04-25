const std = @import("std");
const tensorError = @import("error.zig").tensorError;
const TensorError = @import("error.zig").TensorError;

fn numItems(shape: []usize) usize {
    var total: usize = 1;
    for (shape) |item| {
        total *= item;
    }
    return total;
}

pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T,
        shape: []usize,
        allocator: std.mem.Allocator,
        stride: []usize,

        const Self = @This();

        pub fn format(self: Self, writer: *std.Io.Writer) !void {
            // print as many [ as tensor dims
            for (0..self.shape.len) |_| {
                try writer.writeAll("[");
            }

            for (self.data) |_| {}
            try writer.print("tensor()", .{});
        }

        pub fn ensureShape(data: []T, total: usize) !void {
            if (data.len != total) {
                return tensorError(TensorError.SHAPE_MISMATCH);
            }
        }

        pub fn empty(shape: []usize, allocator: std.mem.Allocator) !Self {
            const total = numItems(shape);
            const data = try allocator.alloc(T, total);
            const stride = try getStride(shape, allocator);
            const shape_copy = try allocator.dupe(usize, shape);
            return Self{ .data = data, .shape = shape_copy, .allocator = allocator, .stride = stride };
        }

        pub fn zeros(shape: []usize, allocator: std.mem.Allocator) !Self {
            const self = try Self.empty(shape, allocator);
            @memset(self.data, 0);
            return self;
        }

        fn getStride(shape: []usize, allocator: std.mem.Allocator) ![]usize {
            const stride = try allocator.alloc(usize, shape.len);

            // init stride is always 1
            stride[shape.len - 1] = 1;

            // then walk backwards from the shape end and multiply each time along the way
            // except for the first dim
            var i: usize = shape.len - 1;
            while (i > 0) {
                i -= 1;
                stride[i] = stride[i + 1] * shape[i + 1];
            }

            return stride;
        }

        pub fn at(self: *Self, indices: []const usize) T {
            // example:
            // shape 3, 4, 5
            // [[[6, 2, 9, 6, 8],
            //   [2, 3, 5, 3, 2],
            //   [7, 3, 1, 4, 2],
            //   [0, 8, 3, 3, 1]],
            //
            //  [[8, 3, 5, 3, 7],
            //   [0, 4, 8, 2, 5],
            //   [9, 8, 2, 7, 0],
            //   [8, 9, 3, 2, 2]],
            //
            //  [[3, 4, 4, 5, 7],
            //   [2, 6, 1, 1, 0],
            //   [3, 5, 7, 1, 2],
            //   [4, 3, 8, 0, 7]]])
            //   stride(20, 5, 1)
            //   flat_index = i * 20 + j * 5 + k * 1
            var flat_index: usize = 0;
            for (indices, 0..) |idx, i| {
                flat_index += idx * self.stride[i];
            }
            return self.data[flat_index];
        }
        pub fn fromSlice(data: []T, shape: []usize, allocator: std.mem.Allocator) !Self {
            const total = numItems(shape);
            try ensureShape(data, total);
            const data_copy = try allocator.dupe(T, data);
            const stride = try getStride(shape, allocator);
            const shape_copy = try allocator.dupe(usize, shape);
            return Self{ .data = data_copy, .shape = shape_copy, .allocator = allocator, .stride = stride };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
            self.allocator.free(self.stride);
        }
    };
}

test "tensor empty has correct shape" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4 };
    var tens = try Tensor(f32).empty(&shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 3), tens.shape[0]);
    try std.testing.expectEqual(@as(usize, 4), tens.shape[1]);
    try std.testing.expectEqual(@as(usize, 12), tens.data.len);
}

test "tensor zeros are all zero" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4 };
    var tens = try Tensor(f32).zeros(&shape, gpa);
    defer tens.deinit();

    for (tens.data) |item| {
        try std.testing.expectEqual(@as(f32, 0), item);
    }
}

test "tensor at returns correct value" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4 };
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var tens = try Tensor(f32).fromSlice(&data, &shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(f32, 1), tens.at(&.{ 0, 0 }));
    try std.testing.expectEqual(@as(f32, 2), tens.at(&.{ 0, 1 }));
    try std.testing.expectEqual(@as(f32, 5), tens.at(&.{ 1, 0 }));
    try std.testing.expectEqual(@as(f32, 12), tens.at(&.{ 2, 3 }));
}

test "tensor fromSlice has correct data" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var data = [_]f32{ 1, 2, 3, 4 };
    var tens = try Tensor(f32).fromSlice(&data, &shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 4), tens.data.len);
    try std.testing.expectEqual(@as(f32, 1), tens.data[0]);
    try std.testing.expectEqual(@as(f32, 4), tens.data[3]);
}

test "tensor dtype f16 zeros" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var tens = try Tensor(f16).zeros(&shape, gpa);
    defer tens.deinit();

    for (tens.data) |item| {
        try std.testing.expectEqual(@as(f16, 0), item);
    }
}

test "tensor stride are correct for 2d" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4 };
    var tens = try Tensor(f32).empty(&shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 4), tens.stride[0]);
    try std.testing.expectEqual(@as(usize, 1), tens.stride[1]);
}

test "tensor stride are correct for 3d" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{ 3, 4, 5 };
    var tens = try Tensor(f32).empty(&shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 20), tens.stride[0]);
    try std.testing.expectEqual(@as(usize, 5), tens.stride[1]);
    try std.testing.expectEqual(@as(usize, 1), tens.stride[2]);
}

test "tensor stride are correct for 1d" {
    const gpa = std.testing.allocator;
    var shape = [_]usize{6};
    var tens = try Tensor(f32).empty(&shape, gpa);
    defer tens.deinit();

    try std.testing.expectEqual(@as(usize, 1), tens.stride[0]);
}
