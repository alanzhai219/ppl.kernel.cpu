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
#include <map>
#include <string>
#include <random>
#include <limits>
#include <cmath>
#include <chrono>
#include <cfloat>

#if defined(__linux__) && defined(PPL_USE_X86_OMP)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <omp.h>
#endif

#include "ppl/kernel/x86/fp32/abs.h"
#include "ppl/common/tensor_shape.h"
#include "simple_flags.h"

Define_bool_opt("--help", Flag_help, false, "show these help information");
Define_string(isa, "auto", "(auto) noarch, sse, avx, auto");
Define_bool(benchmark, false, "(false) run benchmark instead of correctness-only test");
Define_bool(validate, true, "(true) validate output before benchmark loop");
Define_int64(len, 16 * 1024 * 1024, "(16777216) benchmark tensor length");
Define_int32(warm_up, 5, "(5) benchmark warm up iterations");
Define_int32(min_iter, 50, "(50) minimum benchmark iterations");
Define_float(min_second, 1.0f, "(1.0) minimum benchmark duration in seconds");
Define_int32(seed, 20260511, "(20260511) random seed for benchmark input");
Define_int32(num_threads, 1, "(1) number of threads for benchmark when OpenMP is enabled");
Define_bool(core_bind, false, "(false) bind OpenMP worker threads to cores when OpenMP is enabled");

static bool equal_value(const float dst, const float ref)
{
	if (std::isnan(dst) || std::isnan(ref)) {
		return std::isnan(dst) && std::isnan(ref);
	}
	if (std::isinf(dst) || std::isinf(ref)) {
		return dst == ref;
	}
	if (dst == 0.0f && ref == 0.0f) {
		return std::signbit(dst) == std::signbit(ref);
	}
	return dst == ref;
}

static bool test_one_case(
	const std::vector<float> &src,
	const ppl::common::isa_t isa,
	const std::string &isa_name,
	const std::string &case_name)
{
	ppl::common::TensorShape shape;
	shape.Reshape({(int64_t)src.size()});

	std::vector<float> dst(src.size(), 0.0f);
	std::vector<float> ref(src.size(), 0.0f);

	auto rc_ref = ppl::kernel::x86::abs_fp32_ref(&shape, src.data(), ref.data());
	auto rc = ppl::kernel::x86::abs_fp32(isa, &shape, src.data(), dst.data());
	if (rc_ref != ppl::common::RC_SUCCESS || rc != ppl::common::RC_SUCCESS) {
		std::cerr << "run failed, isa=" << isa_name << " case=" << case_name << "\n";
		return false;
	}

	for (size_t i = 0; i < src.size(); ++i) {
		if (!equal_value(dst[i], ref[i])) {
			std::cerr << "mismatch, isa=" << isa_name
					  << ", case=" << case_name
					  << ", idx=" << i
					  << ", src=" << src[i]
					  << ", dst=" << dst[i]
					  << ", ref=" << ref[i] << "\n";
			return false;
		}
	}
	return true;
}

static bool test_corner_cases(const ppl::common::isa_t isa, const std::string &isa_name)
{
	const float inf = std::numeric_limits<float>::infinity();
	const std::vector<std::vector<float>> cases = {
		{-0.0f},
		{0.0f},
		{-1.0f},
		{1.0f},
		{-0.0f, 0.0f, -1.0f, 1.0f},
		{-inf, inf, -123.5f, 456.25f},
		{-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f,
		 -9.0f, 10.0f, -11.0f, 12.0f, -13.0f, 14.0f, -15.0f, 16.0f, -17.0f}
	};

	for (size_t i = 0; i < cases.size(); ++i) {
		if (!test_one_case(cases[i], isa, isa_name, "corner_" + std::to_string(i))) {
			return false;
		}
	}
	return true;
}

static bool test_random_cases(const ppl::common::isa_t isa, const std::string &isa_name)
{
	std::mt19937 rng(20260511);
	std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
	const std::vector<int64_t> lens = {1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65};

	for (size_t case_idx = 0; case_idx < lens.size(); ++case_idx) {
		const int64_t len = lens[case_idx];
		std::vector<float> src(len);
		for (int64_t i = 0; i < len; ++i) {
			src[i] = dist(rng);
		}
		if (!test_one_case(src, isa, isa_name, "random_" + std::to_string(len))) {
			return false;
		}
	}
	return true;
}

static bool benchmark_abs(const ppl::common::isa_t isa, const std::string &isa_name)
{
	if (Flag_len <= 0) {
		std::cerr << "invalid len: " << Flag_len << "\n";
		return false;
	}
	if (Flag_num_threads <= 0) {
		std::cerr << "invalid num_threads: " << Flag_num_threads << "\n";
		return false;
	}

	int32_t num_threads = 1;
#if defined(__linux__) && defined(PPL_USE_X86_OMP)
	omp_set_num_threads(Flag_num_threads);
	num_threads = Flag_num_threads;
	if (Flag_core_bind) {
#pragma omp parallel
		{
#define handle_error_en(en, msg) do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)
			int i = omp_get_thread_num();
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(i, &cpuset);
			if (int s = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
				handle_error_en(s, "pthread_setaffinity_np");
			}
#undef handle_error_en
		}
	}
#else
	if (Flag_num_threads != 1 || Flag_core_bind) {
		std::cerr << "warning: thread control requires OpenMP build support; using single-thread benchmark\n";
	}
#endif

	ppl::common::TensorShape shape;
	shape.Reshape({Flag_len});

	std::mt19937 rng(Flag_seed);
	std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
	std::vector<float> src(Flag_len);
	std::vector<float> dst(Flag_len, 0.0f);
	std::vector<float> ref;
	for (int64_t i = 0; i < Flag_len; ++i) {
		src[i] = dist(rng);
	}

	if (Flag_validate) {
		ref.resize(Flag_len, 0.0f);
		auto rc_ref = ppl::kernel::x86::abs_fp32_ref(&shape, src.data(), ref.data());
		auto rc = ppl::kernel::x86::abs_fp32(isa, &shape, src.data(), dst.data());
		if (rc_ref != ppl::common::RC_SUCCESS || rc != ppl::common::RC_SUCCESS) {
			std::cerr << "benchmark validation run failed, isa=" << isa_name << "\n";
			return false;
		}
		for (int64_t i = 0; i < Flag_len; ++i) {
			if (!equal_value(dst[i], ref[i])) {
				std::cerr << "benchmark validation mismatch, isa=" << isa_name
						  << ", idx=" << i
						  << ", src=" << src[i]
						  << ", dst=" << dst[i]
						  << ", ref=" << ref[i] << "\n";
				return false;
			}
		}
	}

	for (int32_t i = 0; i < Flag_warm_up; ++i) {
		auto rc = ppl::kernel::x86::abs_fp32(isa, &shape, src.data(), dst.data());
		if (rc != ppl::common::RC_SUCCESS) {
			std::cerr << "warm up failed, isa=" << isa_name << "\n";
			return false;
		}
	}

	double total_us = 0.0;
	double min_us = DBL_MAX;
	int32_t iter = 0;
	for (; iter < Flag_min_iter || total_us < Flag_min_second * 1e6; ++iter) {
		auto begin = std::chrono::high_resolution_clock::now();
		auto rc = ppl::kernel::x86::abs_fp32(isa, &shape, src.data(), dst.data());
		auto end = std::chrono::high_resolution_clock::now();
		if (rc != ppl::common::RC_SUCCESS) {
			std::cerr << "benchmark run failed, isa=" << isa_name << "\n";
			return false;
		}
		const double elapsed_us = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - begin).count();
		total_us += elapsed_us;
		min_us = std::min(min_us, elapsed_us);
	}

	const double avg_us = total_us / iter;
	const double bytes = static_cast<double>(Flag_len) * sizeof(float) * 2.0;
	const double max_gbps = bytes / min_us / 1e3;
	const double avg_gbps = bytes / avg_us / 1e3;

	std::cout << "%isa,len,num_threads,min_ms,max_gbps,avg_ms,avg_gbps,iters\n";
	std::cout << isa_name << ","
			  << Flag_len << ","
			  << num_threads << ","
			  << min_us / 1e3 << ","
			  << max_gbps << ","
			  << avg_us / 1e3 << ","
			  << avg_gbps << ","
			  << iter << std::endl;
	return true;
}

int main(int argc, char **argv)
{
	simple_flags::parse_args(argc, argv);
	if (Flag_help) {
		simple_flags::print_args_info();
		return 0;
	}

	std::map<std::string, ppl::common::isa_t> isa_map = {
		{"noarch", ppl::common::ISA_UNKNOWN},
		{"sse", ppl::common::ISA_X86_SSE},
		{"avx", ppl::common::ISA_X86_AVX},
	};

	ppl::common::isa_t isa = ppl::common::ISA_UNKNOWN;
	if (Flag_isa == "auto") {
		isa = ppl::common::GetCpuISA();
	} else {
		auto it = isa_map.find(Flag_isa);
		if (it == isa_map.end()) {
			std::cerr << "unsupported isa: " << Flag_isa << "\n";
			return -1;
		}
		isa = it->second;
	}

	if (Flag_benchmark) {
		return benchmark_abs(isa, Flag_isa) ? 0 : -1;
	}

	if (!test_corner_cases(isa, Flag_isa)) {
		return -1;
	}
	if (!test_random_cases(isa, Flag_isa)) {
		return -1;
	}

	std::cout << "test_abs pass" << std::endl;
	return 0;
}
