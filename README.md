# ppl.kernel.cpu

`ppl.kernel.cpu` 是 OpenPPL 的 CPU kernel 仓库，包含 x86、ARM、RISC-V 等后端的算子实现、头文件和测试程序。

常用目录：

- `include/`: 对外头文件
- `src/`: kernel 实现
- `test/`: 测试与 benchmark
- `cmake/`: 各架构构建脚本

## 编译

x86 默认不会自动打开，推荐使用单独构建目录：

```bash
cd //ppl.kernel.cpu
mkdir -p build
cd build
cmake -DPPLNN_USE_X86_64=ON -DPPLNN_BUILD_TESTS=ON -DPPLNNX86Kernel_HOLD_DEPS=ON ..
cmake --build . -j4
```

只编译某个测试目标：

```bash
cmake --build . --target test_abs -j4
```

常用开关：

- `PPLNN_USE_X86_64=ON`: 打开 x86 kernel
- `PPLNN_BUILD_TESTS=ON`: 生成测试目标
- `PPLNNX86Kernel_HOLD_DEPS=ON`: 冻结当前依赖
- `PPLNN_USE_OPENMP=ON`: 打开 OpenMP，多线程 benchmark 需要它

## 如何新加算子

以 x86 fp32 为例，推荐沿用这组结构：

- `include/ppl/kernel/x86/fp32/my_op.h`
- `src/ppl/kernel/x86/fp32/my_op/my_op_fp32.cpp`
- `src/ppl/kernel/x86/fp32/my_op/my_op_fp32_sse.cpp`
- `src/ppl/kernel/x86/fp32/my_op/my_op_fp32_avx.cpp`
- `test/test_my_op.cpp`

约定：

- `*_fp32.cpp`: 参考实现和 ISA 分发
- `*_sse.cpp` / `*_avx.cpp` / `*_fma.cpp` / `*_avx512.cpp`: ISA 特化实现
- 统一入口先判断 `isa`，再调用对应实现，否则退回 `ref`

x86 源文件在 `cmake/x86.cmake` 中通过 `GLOB_RECURSE` 自动收集，通常不需要手动登记源码文件。

## 如何测试算子

当前仓库是“一个测试目标对应一个可执行文件”，不是 gtest 风格。

新增测试时：

1. 在 `test/` 下增加 `test_my_op.cpp`
2. 在 `cmake/x86.cmake` 的 `PPLKERNELX86_TESTS` 里追加 `test_my_op`
3. 编译并运行

```bash
cd //ppl.kernel.cpu/build
cmake --build . --target test_my_op -j4
./test_my_op
```

注意：`test/` 里的程序使用 `simple_flags`，参数是单横线形式，例如 `-isa=avx`。

`test_abs` 的 correctness 用法：

```bash
./test_abs
./test_abs -isa=noarch
./test_abs -isa=sse
./test_abs -isa=avx
```

## 如何测试性能

现有 `test_gemm`、`test_conv2d`、`test_pd_conv2d` 本身就带 benchmark 参数，例如：

```bash
./test_gemm -cfg=/path/to/gemm.cfg -isa=fma -warm_up=5 -min_iter=50 -min_second=2
./test_conv2d -cfg=/path/to/conv.cfg -algo=n16cx_gemm_direct_fp32_fma -warm_up=5 -min_iter=20 -min_second=2
```

`test_abs` 现在也支持 benchmark 模式：

```bash
./test_abs -benchmark=true -validate=false -isa=avx -len=16777216 -warm_up=5 -min_iter=50 -min_second=2 -num_threads=1
```

如果构建时打开了 OpenMP，还可以控制线程数和绑核：

```bash
./test_abs -benchmark=true -validate=false -isa=avx -len=16777216 -warm_up=5 -min_iter=50 -min_second=2 -num_threads=8 -core_bind=true
```

`test_abs` benchmark 常用参数：

- `-benchmark`: 打开 benchmark 模式
- `-validate`: benchmark 前先做一次正确性校验
- `-isa`: `noarch`、`sse`、`avx` 或 `auto`
- `-len`: 输入元素个数
- `-warm_up`: 预热次数
- `-min_iter`: 最少迭代次数
- `-min_second`: 最短测试时长
- `-num_threads`: OpenMP 线程数
- `-core_bind`: 是否绑核

输出字段包括 `min_ms`、`avg_ms`、`max_gbps`、`avg_gbps` 和 `num_threads`。

对 `abs` 这类简单 unary 算子，更建议看 `GB/s` 而不是 `GFLOPS`。

如需补充底层性能指标，可以直接配合 `perf stat`：

```bash
perf stat ./test_abs -isa=avx
```
