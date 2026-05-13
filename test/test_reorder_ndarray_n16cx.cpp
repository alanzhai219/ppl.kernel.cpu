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
#include <cstdint>

#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/common/tensor_shape.h"
#include "simple_flags.h"

Define_bool_opt("--help", Flag_help, false, "show these help information");

struct reorder_case {
    std::vector<int64_t> dims;
    const char *name;
};

static int64_t round_up_to_blk(const int64_t value, const int64_t blk)
{
    return (value + blk - 1) / blk * blk;
}

static void fill_src(std::vector<float> *src)
{
    for (size_t i = 0; i < src->size(); ++i) {
        (*src)[i] = static_cast<float>(i % 251) - 125.0f;
    }
}

static void reference_reorder_ndarray_n16cx(
    const ppl::common::TensorShape &src_shape,
    const std::vector<float> &src,
    std::vector<float> *dst)
{
    const int64_t batch = src_shape.GetDim(0);
    const int64_t channels = src_shape.GetDim(1);
    const int64_t x_dim = src_shape.CalcElementsExcludingPadding() / batch / channels;
    const int64_t c_blk = 16;
    const int64_t padded_c = round_up_to_blk(channels, c_blk);

    std::fill(dst->begin(), dst->end(), 0.0f);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t c = 0; c < channels; ++c) {
            const int64_t c_outer = c / c_blk * c_blk;
            const int64_t c_inner = c % c_blk;
            for (int64_t x = 0; x < x_dim; ++x) {
                const int64_t src_idx = (b * channels + c) * x_dim + x;
                const int64_t dst_idx = b * padded_c * x_dim + c_outer * x_dim + x * c_blk + c_inner;
                (*dst)[dst_idx] = src[src_idx];
            }
        }
    }
}

static bool run_one_case(const reorder_case &tc)
{
    ppl::common::TensorShape src_shape;
    src_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
    src_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
    src_shape.Reshape(tc.dims);

    ppl::common::TensorShape dst_shape = src_shape;
    dst_shape.SetDataFormat(ppl::common::DATAFORMAT_N16CX);

    std::vector<float> src(src_shape.CalcElementsIncludingPadding());
    fill_src(&src);

    std::vector<float> dst(dst_shape.CalcElementsIncludingPadding(), -1.0f);
    std::vector<float> ref(dst_shape.CalcElementsIncludingPadding(), 0.0f);
    std::vector<float> roundtrip(src_shape.CalcElementsIncludingPadding(), 0.0f);

    reference_reorder_ndarray_n16cx(src_shape, src, &ref);

    auto rc = ppl::kernel::x86::reorder_ndarray_n16cx_fp32_avx(&src_shape, src.data(), dst.data());
    if (rc != ppl::common::RC_SUCCESS) {
        std::cerr << "reorder_ndarray_n16cx_fp32_avx failed for case " << tc.name << "\n";
        return false;
    }

    for (size_t i = 0; i < dst.size(); ++i) {
        if (dst[i] != ref[i]) {
            std::cerr << "packed mismatch in case " << tc.name
                      << ", idx=" << i
                      << ", dst=" << dst[i]
                      << ", ref=" << ref[i] << "\n";
            return false;
        }
    }

    rc = ppl::kernel::x86::reorder_n16cx_ndarray_fp32_avx(&dst_shape, dst.data(), roundtrip.data());
    if (rc != ppl::common::RC_SUCCESS) {
        std::cerr << "reorder_n16cx_ndarray_fp32_avx failed for case " << tc.name << "\n";
        return false;
    }

    for (size_t i = 0; i < src.size(); ++i) {
        if (roundtrip[i] != src[i]) {
            std::cerr << "roundtrip mismatch in case " << tc.name
                      << ", idx=" << i
                      << ", dst=" << roundtrip[i]
                      << ", ref=" << src[i] << "\n";
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

    const std::vector<reorder_case> cases = {
        {{1, 3, 4, 5}, "nchw_c3_hw20"},
        {{2, 16, 3, 7}, "nchw_c16_exact_block"},
        {{1, 17, 2, 3, 5}, "ndhw_c17_tail_block"},
        {{3, 31, 9}, "ncx_c31_x9"},
    };

    for (const auto &tc : cases) {
        if (!run_one_case(tc)) {
            return 1;
        }
    }

    std::cout << "all reorder ndarray->n16cx tests passed\n";
    return 0;
}