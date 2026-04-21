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

        const Self = @This();

        pub fn ensureShape(data: []T, total: usize) !void {
            if (data.len != total) {
                return tensorError(TensorError.SHAPE_MISMATCH);
            }
        }

        pub fn empty(shape: []usize, allocator: std.mem.Allocator) !Self {
            const total = numItems(shape);
            const data = try allocator.alloc(T, total);
            const shape_copy = try allocator.dupe(usize, shape);
            return Self{ .data = data, .shape = shape_copy, .allocator = allocator };
        }

        pub fn zeros(shape: []usize, allocator: std.mem.Allocator) !Self {
            const self = try Self.empty(shape, allocator);
            @memset(self.data, 0);
            return self;
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
                var stride: usize = 1;
                for (self.shape[i + 1 ..]) |dim| {
                    stride *= dim;
                }
                flat_index += idx * stride;
            }
            return self.data[flat_index];
        }
        pub fn fromSlice(data: []T, shape: []usize, allocator: std.mem.Allocator) !Self {
            // TODO: data/shape mismatch err
            const total = numItems(shape);
            try ensureShape(data, total);
            const data_copy = try allocator.dupe(T, data);
            const shape_copy = try allocator.dupe(usize, shape);
            return Self{ .data = data_copy, .shape = shape_copy, .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
        }
    };
}
