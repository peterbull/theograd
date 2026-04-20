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
    std.debug.print("val: {any}", .{val});
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(i32) = .empty;
    defer list.deinit(gpa); // Try commenting this out and see if zig detects the memory leak!
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}

test "fuzz example" {
    const Context = struct {
        fn testOne(context: @This(), input: []const u8) anyerror!void {
            _ = context;
            // Try passing `--fuzz` to `zig build test` and see if it manages to fail this test case!
            try std.testing.expect(!std.mem.eql(u8, "canyoufindme", input));
        }
    };
    try std.testing.fuzz(Context{}, Context.testOne, .{});
}
