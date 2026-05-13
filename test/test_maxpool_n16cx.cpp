// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include "ppl/kernel/x86/fp32/maxpool2d.h"
#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/common/tensor_shape.h"
#include "simple_flags.h"

Define_bool_opt("--help", Flag_help, false, "show these help information");
Define_int32(seed, 20260513, "(20260513) random seed");

struct maxpool2d_case {
    int64_t batch;
    int64_t channels;
    int64_t src_h;
    int64_t src_w;
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride_h;
    int64_t stride_w;
    int64_t pad_h;
    int64_t pad_w;
    const char *name;
};

static bool equal_value(float a, float b)
{
    if (std::isnan(a) || std::isnan(b)) {
        return std::isnan(a) && std::isnan(b);
    }
    return a == b;
}

static int64_t calc_out_dim(int64_t in, int64_t kernel, int64_t stride, int64_t pad)
{
    return (in + 2 * pad - kernel) / stride + 1;
}

static bool run_one_case(const maxpool2d_case &tc, std::mt19937 &rng)
{
    const int64_t dst_h = calc_out_dim(tc.src_h, tc.kernel_h, tc.stride_h, tc.pad_h);
    const int64_t dst_w = calc_out_dim(tc.src_w, tc.kernel_w, tc.stride_w, tc.pad_w);

    ppl::common::TensorShape src_shape;
    src_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
    src_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    src_shape.Reshape({tc.batch, tc.channels, tc.src_h, tc.src_w});

    ppl::common::TensorShape src_n16cx_shape = src_shape;
    src_n16cx_shape.SetDataFormat(ppl::common::DATAFORMAT_N16CX);

    ppl::common::TensorShape dst_shape;
    dst_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
    dst_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    dst_shape.Reshape({tc.batch, tc.channels, dst_h, dst_w});

    ppl::common::TensorShape dst_n16cx_shape = dst_shape;
    dst_n16cx_shape.SetDataFormat(ppl::common::DATAFORMAT_N16CX);

    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> src(src_shape.CalcElementsIncludingPadding());
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = dist(rng);
    }

    std::vector<float> src_n16cx(src_n16cx_shape.CalcElementsIncludingPadding(), 0.0f);
    std::vector<float> dst_ref(dst_shape.CalcElementsIncludingPadding(), 0.0f);
    std::vector<float> dst_n16cx(dst_n16cx_shape.CalcElementsIncludingPadding(), 0.0f);
    std::vector<float> dst(dst_shape.CalcElementsIncludingPadding(), 0.0f);

    auto rc = ppl::kernel::x86::reorder_ndarray_n16cx_fp32_avx(&src_shape, src.data(), src_n16cx.data());
    if (rc != ppl::common::RC_SUCCESS) {
        std::cerr << "reorder_ndarray_n16cx failed for case " << tc.name << "\n";
        return false;
    }

    rc = ppl::kernel::x86::maxpool2d_ndarray_normal_fp32(
        &src_shape,
        &dst_shape,
        src.data(),
        tc.kernel_h,
        tc.kernel_w,
        tc.stride_h,
        tc.stride_w,
        tc.pad_h,
        tc.pad_w,
        dst_ref.data());
    if (rc != ppl::common::RC_SUCCESS) {
        std::cerr << "maxpool2d_ndarray_normal_fp32 failed for case " << tc.name << "\n";
        return false;
    }

    rc = ppl::kernel::x86::maxpool2d_n16cx_blk1x8_fp32_avx(
        &src_n16cx_shape,
        &dst_n16cx_shape,
        src_n16cx.data(),
        tc.kernel_h,
        tc.kernel_w,
        tc.stride_h,
        tc.stride_w,
        tc.pad_h,
        tc.pad_w,
        dst_n16cx.data());
    if (rc != ppl::common::RC_SUCCESS) {
        std::cerr << "maxpool2d_n16cx_blk1x8_fp32_avx failed for case " << tc.name << "\n";
        return false;
    }

    rc = ppl::kernel::x86::reorder_n16cx_ndarray_fp32_avx(&dst_n16cx_shape, dst_n16cx.data(), dst.data());
    if (rc != ppl::common::RC_SUCCESS) {
        std::cerr << "reorder_n16cx_ndarray failed for case " << tc.name << "\n";
        return false;
    }

    for (int64_t i = 0; i < dst_shape.CalcElementsExcludingPadding(); ++i) {
        if (!equal_value(dst[i], dst_ref[i])) {
            std::cerr << "mismatch in case " << tc.name
                      << ", idx=" << i
                      << ", dst=" << dst[i]
                      << ", ref=" << dst_ref[i] << "\n";
            return false;
        }
    }

    std::cout << "pass: " << tc.name << "\n";
    return true;
}

int main(int argc, char **argv)
{
    simple_flags::parse_args(argc, argv);
    if (Flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    const std::vector<maxpool2d_case> cases = {
        {1, 3, 4, 5, 2, 2, 2, 2, 0, 0, "basic_c3_stride2"},
        {1, 13, 5, 6, 3, 3, 1, 1, 1, 1, "border_c13_pad1"},
        {1, 16, 3, 9, 2, 3, 1, 2, 0, 1, "exact_block_c16_mixed_stride"},
        {2, 17, 7, 8, 3, 3, 2, 2, 1, 1, "batch2_c17_stride2"},
    };

    std::mt19937 rng(Flag_seed);
    for (const auto &tc : cases) {
        if (!run_one_case(tc, rng)) {
            return 1;
        }
    }

    std::cout << "all maxpool n16cx tests passed\n";
    return 0;
}