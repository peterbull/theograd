const std = @import("std");
const graditude = @import("graditude");
const Tensor = @import("tensor.zig").Tensor;

// pub fn matmul(x: Tensor, y: Tensor) Tensor {
//     unreachable;
// }
pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var arr = [_]usize{
        3,
        4,
    };

    var F32Tens = try Tensor(f32).empty(&arr, gpa);
    defer F32Tens.deinit();

    std.debug.print("break\n", .{});

    _ = &F32Tens;
    std.debug.print("break\n", .{});
    for (F32Tens.data) |item| {
        _ = item * 4;
        std.debug.print("item: {any}\n", .{item});
    }

    std.debug.print("tens data: {any}\n", .{F32Tens.data});
    std.debug.print("break\n", .{});

    var F16Zeros = try Tensor(f16).zeros(&arr, gpa);
    defer F16Zeros.deinit();

    const val = F16Zeros.at(&.{ 0, 0 });

    std.debug.print("zeros data: {any}\n", .{F16Zeros.data});
    std.debug.print("break\n", .{});
    std.debug.print("val: {any}\n", .{val});

    var slice_arr = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    var F32FromSlice = try Tensor(f32).fromSlice(&slice_arr, &arr, gpa);
    defer F32FromSlice.deinit();

    std.debug.print("fromslice data: {any}\n", .{F32FromSlice.data});
    std.debug.print("break\n", .{});
    std.debug.print("stride data: {any}\n", .{F32FromSlice.stride});
    std.debug.print("break\n", .{});
}
