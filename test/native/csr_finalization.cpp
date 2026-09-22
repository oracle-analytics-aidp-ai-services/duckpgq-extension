// Real CSR owners, tuple writers and DuckDB budget context. No worker substitute.
#include "duckpgq_state.hpp"
#include "duckdb.hpp"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

using namespace duckdb;
static_assert(!std::is_copy_constructible<CSR::EdgeWrite>::value, "a reservation must have one owner");
static_assert(std::is_move_constructible<CSR::EdgeWrite>::value, "a reservation can transfer ownership");

static void require(bool ok, const char *message) {
	if (!ok)
		throw std::runtime_error(message);
}
template <class Error, class Operation>
static void refuses(Operation operation, const char *message) {
	try {
		operation();
	} catch (const Error &) {
		return;
	} catch (...) {
		throw std::runtime_error(std::string(message) + ": wrong exception class");
	}
	throw std::runtime_error(std::string(message) + ": unexpectedly accepted");
}
template <class Container>
static void sequence(const Container &actual, std::initializer_list<int64_t> expected) {
	require(actual.size() == expected.size(), "tuple sequence length changed");
	size_t i = 0;
	for (const auto &value : expected)
		require(actual[i++] == value, "tuple sequence changed");
}
static uint64_t bits(double value) {
	uint64_t out;
	static_assert(sizeof(out) == sizeof(value), "binary64 weights required");
	std::memcpy(&out, &value, sizeof(out));
	return out;
}
static double from_bits(uint64_t value) {
	double out;
	std::memcpy(&out, &value, sizeof(out));
	return out;
}
static void weight_bits(const shared_ptr<CSR> &csr, std::initializer_list<uint64_t> expected) {
	require(csr->w_double.size() == expected.size(), "double weight length changed");
	size_t i = 0;
	for (auto value : expected)
		require(bits(csr->w_double[i++]) == value, "double weight bits changed");
}
static void offsets(const shared_ptr<CSR> &csr, std::initializer_list<int64_t> expected) {
	require(csr->vsize == expected.size(), "vertex padding shape changed");
	size_t i = 0;
	for (auto value : expected)
		require(csr->v[i++].load() == value, "CSR offset or padding changed");
}
static shared_ptr<CSR> graph(DuckPGQState &state, int id, std::initializer_list<int64_t> degrees,
                             CSRWeightMode mode = CSRWeightMode::Unweighted) {
	auto csr = state.InitializeVertex(id, static_cast<int64_t>(degrees.size()));
	int64_t source = 0, total = 0;
	for (auto degree : degrees) {
		csr->SetVertexDegree(source++, degree);
		total += degree;
	}
	state.InitializeEdges(id, source, total);
	if (mode == CSRWeightMode::Int64)
		state.InitializeWeights(id, total, PhysicalType::INT64);
	if (mode == CSRWeightMode::Double)
		state.InitializeWeights(id, total, PhysicalType::DOUBLE);
	return csr;
}
static void unweighted(Connection &connection) {
	DuckPGQState state;
	auto csr = graph(state, 1, {6, 1, 0, 0});
	const int64_t targets[] = {2, 1, 1, 1, 0, 2};
	const int64_t ids[] = {8, 7, 7, 2, -1, 3};
	for (int i = 0; i < 6; ++i)
		CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted).Commit(targets[i], ids[i]);
	CSR::BeginEdgeWrite(csr, 1, CSRWeightMode::Unweighted).Commit(0, 9);
	csr->FinalizeForRead(*connection.context);
	offsets(csr, {0, 6, 7, 7, 7, 7});
	sequence(csr->e, {int64_t {0}, 1, 1, 1, 2, 2, 0});
	sequence(csr->edge_ids, {int64_t {-1}, 2, 7, 7, 3, 8, 9});
	require(csr->w.empty() && csr->w_double.empty(), "unweighted graph gained weights");
}
static void integer_weights(Connection &connection) {
	DuckPGQState state;
	auto csr = graph(state, 2, {8, 1, 0, 0}, CSRWeightMode::Int64);
	const int64_t targets[] = {2, 1, 1, 1, 1, 1, 0, 2};
	const int64_t ids[] = {8, 7, 7, 7, 7, 7, -1, 3};
	const int64_t weights[] = {30, 2, 1, 1, std::numeric_limits<int64_t>::min(), 0, -4, 9};
	for (int i = 0; i < 8; ++i)
		CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Int64).Commit(targets[i], ids[i], weights[i]);
	CSR::BeginEdgeWrite(csr, 1, CSRWeightMode::Int64).Commit(0, 9, int64_t {0});
	csr->FinalizeForRead(*connection.context);
	offsets(csr, {0, 8, 9, 9, 9, 9});
	sequence(csr->e, {int64_t {0}, 1, 1, 1, 1, 1, 2, 2, 0});
	sequence(csr->edge_ids, {int64_t {-1}, 7, 7, 7, 7, 7, 3, 8, 9});
	sequence(csr->w, {int64_t {-4}, 0, 1, 1, 2, std::numeric_limits<int64_t>::min(), 9, 30, 0});
	require(csr->w_double.empty(), "integer graph gained double weights");
}
static void double_weights(Connection &connection) {
	DuckPGQState state;
	auto csr = graph(state, 3, {10, 1, 0, 0}, CSRWeightMode::Double);
	const int64_t targets[] = {2, 1, 1, 1, 1, 1, 1, 1, 0, 2};
	const int64_t ids[] = {8, 7, 7, 7, 7, 7, 7, 7, -1, 3};
	const uint64_t weights[] = {0x400c000000000000ULL, 0x8000000000000000ULL, 0x3ff0000000000001ULL,
	                            0x7ff8000000000002ULL, 0x3ff0000000000000ULL, 0,
	                            0x3ff0000000000000ULL, 0x7ff8000000000001ULL, 0xc000000000000000ULL,
	                            0x4022000000000000ULL};
	for (int i = 0; i < 10; ++i)
		CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Double).Commit(targets[i], ids[i], from_bits(weights[i]));
	CSR::BeginEdgeWrite(csr, 1, CSRWeightMode::Double).Commit(0, 9, 0.0);
	csr->FinalizeForRead(*connection.context);
	offsets(csr, {0, 10, 11, 11, 11, 11});
	sequence(csr->e, {int64_t {0}, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0});
	sequence(csr->edge_ids, {int64_t {-1}, 7, 7, 7, 7, 7, 7, 7, 3, 8, 9});
	weight_bits(csr, {0xc000000000000000ULL, 0, 0x3ff0000000000000ULL, 0x3ff0000000000000ULL, 0x3ff0000000000001ULL,
	                  0x7ff8000000000001ULL, 0x7ff8000000000002ULL, 0x8000000000000000ULL, 0x4022000000000000ULL,
	                  0x400c000000000000ULL, 0});
	require(csr->w.empty(), "double graph gained integer weights");
}
static void empty_and_isolated(Connection &connection) {
	for (auto mode : {CSRWeightMode::Unweighted, CSRWeightMode::Int64, CSRWeightMode::Double}) {
		DuckPGQState state;
		auto empty = graph(state, 1, {}, mode);
		empty->FinalizeForRead(*connection.context);
		offsets(empty, {0, 0});
		require(empty->e.empty() && empty->edge_ids.empty() && empty->w.empty() && empty->w_double.empty(),
		        "empty graph gained data");
		require(empty->initialized_w.load() == (mode != CSRWeightMode::Unweighted),
		        "empty declared weight mode was lost");
		auto isolated = graph(state, 2, {0, 0, 0}, mode);
		isolated->FinalizeForRead(*connection.context);
		offsets(isolated, {0, 0, 0, 0, 0});
		require(isolated->e.empty() && isolated->edge_ids.empty(), "isolated graph gained edges");
	}
}
static void concurrent_finalize(Connection &connection) {
	DuckPGQState state;
	auto csr = graph(state, 1, {8, 0});
	for (int64_t i = 7; i >= 0; --i)
		CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted).Commit(1, i);
	std::atomic<bool> start {false}, failed {false};
	std::vector<std::thread> threads;
	for (int i = 0; i < 8; ++i)
		threads.emplace_back([&] {
			while (!start.load())
				std::this_thread::yield();
			try {
				csr->FinalizeForRead(*connection.context);
				csr->FinalizeForRead(*connection.context);
			} catch (...) {
				failed = true;
			}
		});
	start = true;
	for (auto &thread : threads)
		thread.join();
	require(!failed.load(), "concurrent or repeated finalization failed");
	offsets(csr, {0, 8, 8, 8});
	sequence(csr->edge_ids, {int64_t {0}, 1, 2, 3, 4, 5, 6, 7});
}
static void concurrent_producers(Connection &connection) {
	DuckPGQState state;
	auto csr = graph(state, 1, {8, 0}, CSRWeightMode::Int64);
	std::atomic<bool> start {false}, failed {false};
	std::vector<std::thread> threads;
	for (int64_t i = 0; i < 8; ++i)
		threads.emplace_back([&, i] {
			while (!start.load())
				std::this_thread::yield();
			try {
				CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Int64).Commit(1, 7 - i, int64_t {70 - 10 * i});
			} catch (...) {
				failed = true;
			}
		});
	start = true;
	for (auto &thread : threads)
		thread.join();
	require(!failed.load(), "concurrent producer failed");
	csr->FinalizeForRead(*connection.context);
	sequence(csr->edge_ids, {int64_t {0}, 1, 2, 3, 4, 5, 6, 7});
	sequence(csr->w, {int64_t {0}, 10, 20, 30, 40, 50, 60, 70});
	offsets(csr, {0, 8, 8, 8});
}
static void retained_owners(Connection &connection) {
	for (bool query_end : {false, true}) {
		DuckPGQState state;
		auto csr = graph(state, 1, {1, 0});
		auto lease = CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted);
		auto retained = state.GetCSR(1);
		if (query_end) {
			state.ScheduleDelete(1);
			state.QueryEnd();
		} else {
			require(state.DeleteCSR(1), "owner was not removed");
		}
		auto replacement = state.InitializeVertex(1, 2);
		require(replacement.get() != retained.get(), "registry replacement reused retained owner");
		lease.Commit(1, 17);
		retained->FinalizeForRead(*connection.context);
		sequence(retained->e, {int64_t {1}});
		sequence(retained->edge_ids, {int64_t {17}});
		require(state.GetCSR(1).get() == replacement.get(), "retained work changed registry replacement");
	}
}
static void incomplete_writes(Connection &connection) {
	DuckPGQState state;
	auto vertex_only = state.InitializeVertex(1, 2);
	refuses<ConstraintException>([&] { vertex_only->FinalizeForRead(*connection.context); },
	                             "uninitialized edge phase");
	auto csr = graph(state, 2, {1, 0});
	refuses<ConstraintException>([&] { csr->FinalizeForRead(*connection.context); }, "missing edge write");
	auto pending = CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted);
	refuses<ConstraintException>([&] { csr->FinalizeForRead(*connection.context); }, "reserved but unpublished edge");
	auto transferred = std::move(pending);
	refuses<ConstraintException>([&] { pending.Commit(1, 4); }, "moved reservation cannot publish");
	transferred.Commit(1, 4);
	csr->FinalizeForRead(*connection.context);
	auto weighted = graph(state, 3, {1, 0}, CSRWeightMode::Double);
	auto weight_pending = CSR::BeginEdgeWrite(weighted, 0, CSRWeightMode::Double);
	require(weighted->initialized_w.load(), "real weight storage must exist");
	refuses<ConstraintException>([&] { weighted->FinalizeForRead(*connection.context); },
	                             "allocated but unfinished weight");
	weight_pending.Commit(1, 4, 3.25);
	weighted->FinalizeForRead(*connection.context);
	weight_bits(weighted, {0x400a000000000000ULL});
}
static void abandoned_and_single_commit(Connection &connection) {
	DuckPGQState state;
	auto csr = graph(state, 1, {1, 0});
	{ auto abandoned = CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted); }
	refuses<ConstraintException>([&] { csr->FinalizeForRead(*connection.context); }, "abandoned reservation");
	auto complete = graph(state, 2, {1, 0});
	auto writer = CSR::BeginEdgeWrite(complete, 0, CSRWeightMode::Unweighted);
	writer.Commit(1, 4);
	refuses<ConstraintException>([&] { writer.Commit(1, 4); }, "double completion");
}
static void late_writes(Connection &connection) {
	DuckPGQState state;
	auto csr = graph(state, 1, {1, 0});
	CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted).Commit(1, 4);
	csr->FinalizeForRead(*connection.context);
	refuses<ConstraintException>([&] { csr->SetVertexDegree(0, 1); }, "late vertex update");
	refuses<ConstraintException>([&] { CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted); },
	                             "late edge reservation");
	refuses<ConstraintException>([&] { state.InitializeEdges(1, 2, 1); }, "late edge initialization");
	refuses<ConstraintException>([&] { state.InitializeWeights(1, 1, PhysicalType::INT64); },
	                             "late weight initialization");
	offsets(csr, {0, 1, 1, 1});
	sequence(csr->e, {int64_t {1}});
	sequence(csr->edge_ids, {int64_t {4}});
}
static void overreservation_and_modes(Connection &connection) {
	DuckPGQState state;
	auto csr = graph(state, 1, {1, 0});
	auto pending = CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted);
	refuses<ConstraintException>([&] { CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted); }, "overreservation");
	pending.Commit(1, 1);
	csr->FinalizeForRead(*connection.context);
	sequence(csr->e, {int64_t {1}});
	sequence(csr->edge_ids, {int64_t {1}});
	auto unweighted = graph(state, 2, {1, 0});
	refuses<ConstraintException>([&] { CSR::BeginEdgeWrite(unweighted, 0, CSRWeightMode::Int64); },
	                             "missing weight initialization");
	state.InitializeWeights(2, 1, PhysicalType::INT64);
	CSR::BeginEdgeWrite(unweighted, 0, CSRWeightMode::Int64).Commit(1, 2, int64_t {3});
	unweighted->FinalizeForRead(*connection.context);
	sequence(unweighted->w, {int64_t {3}});
	auto weighted = graph(state, 3, {1, 0}, CSRWeightMode::Int64);
	refuses<ConstraintException>([&] { CSR::BeginEdgeWrite(weighted, 0, CSRWeightMode::Double); },
	                             "mixed reservation mode");
	auto wrong_commit = CSR::BeginEdgeWrite(weighted, 0, CSRWeightMode::Int64);
	refuses<ConstraintException>([&] { wrong_commit.Commit(1, 1, 1.0); }, "mixed commit mode");
}
static void invalid_inputs_and_shapes(Connection &connection) {
	DuckPGQState state;
	refuses<InvalidInputException>([&] { state.InitializeVertex(1, -1); }, "negative vertex count");
	refuses<InvalidInputException>([&] { state.InitializeVertex(1, std::numeric_limits<int64_t>::max()); },
	                               "vertex allocation overflow");
	auto vertices = state.InitializeVertex(2, 2);
	refuses<InvalidInputException>([&] { vertices->SetVertexDegree(-1, 0); }, "negative source");
	refuses<InvalidInputException>([&] { vertices->SetVertexDegree(2, 0); }, "source outside vertices");
	refuses<InvalidInputException>([&] { vertices->SetVertexDegree(0, -1); }, "negative degree");
	refuses<InvalidInputException>([&] { state.InitializeEdges(2, 3, 0); }, "edge dimensions");
	auto bad_writer = graph(state, 3, {1, 0});
	refuses<InvalidInputException>([&] { CSR::BeginEdgeWrite(bad_writer, -1, CSRWeightMode::Unweighted); },
	                               "invalid reservation source");
	refuses<InvalidInputException>([&] { CSR::BeginEdgeWrite(bad_writer, 0, static_cast<CSRWeightMode>(99)); },
	                               "invalid weight enum");
	auto bad_target = CSR::BeginEdgeWrite(bad_writer, 0, CSRWeightMode::Unweighted);
	refuses<InvalidInputException>([&] { bad_target.Commit(2, 1); }, "invalid target");
	graph(state, 4, {1, 0});
	refuses<InvalidInputException>([&] { state.InitializeWeights(4, 0, PhysicalType::INT64); }, "weight dimensions");
	graph(state, 5, {1, 0});
	refuses<NotImplementedException>([&] { state.InitializeWeights(5, 1, PhysicalType::INT32); },
	                                 "unsupported weight storage type");
	for (int shape = 0; shape < 4; ++shape) {
		auto csr = graph(state, 10 + shape, {1, 0}, CSRWeightMode::Double);
		CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Double).Commit(1, 4, 2.0);
		if (shape == 0)
			csr->v[1] = 2;
		if (shape == 1)
			csr->e[0] = 2;
		if (shape == 2)
			csr->edge_ids.clear();
		if (shape == 3)
			csr->w_double.clear();
		refuses<ConstraintException>([&] { csr->FinalizeForRead(*connection.context); },
		                             "invalid finalized storage shape");
	}
}
static void failure_cleanup(Connection &connection) {
	DuckPGQState state;
	auto csr = graph(state, 1, {1, 0});
	auto pending = CSR::BeginEdgeWrite(csr, 0, CSRWeightMode::Unweighted);
	refuses<ConstraintException>([&] { state.FinalizeCSRForRead(1, csr, *connection.context); },
	                             "state-bound incomplete reader");
	state.QueryEnd();
	refuses<ConstraintException>([&] { state.GetCSR(1); }, "failed owner must leave registry at QueryEnd");
	pending.Commit(1, 4);
	csr->FinalizeForRead(*connection.context);
	sequence(csr->e, {int64_t {1}});
}
static void replacement_cleanup(Connection &connection) {
	DuckPGQState state;
	auto old = graph(state, 1, {1, 0});
	require(state.DeleteCSR(1), "old owner must exist");
	auto replacement = graph(state, 1, {0, 0});
	refuses<ConstraintException>([&] { state.FinalizeCSRForRead(1, old, *connection.context); },
	                             "old incomplete owner");
	state.QueryEnd();
	require(state.GetCSR(1).get() == replacement.get(), "old failure deleted a replacement owner");
	replacement->FinalizeForRead(*connection.context);
	offsets(replacement, {0, 0, 0, 0});
}
static void budget_refusals(Connection &connection) {
	auto limited = connection.Query("SET memory_limit='1MB'");
	require(!limited->HasError(), "real DuckDB memory limit setup failed");
	const int64_t n = 131072;
	DuckPGQState state;
	auto first = state.InitializeVertex(1, 2);
	first->SetVertexDegree(0, n);
	first->SetVertexDegree(1, 0);
	refuses<OutOfMemoryException>([&] { state.InitializeEdges(1, 2, n, connection.context.get()); },
	                              "edge allocation budget");
	state.QueryEnd();
	refuses<ConstraintException>([&] { state.GetCSR(1); }, "failed allocation owner must be cleaned");
	auto csr = graph(state, 2, {n, 0});
	refuses<OutOfMemoryException>([&] { state.InitializeWeights(2, n, PhysicalType::INT64, connection.context.get()); },
	                              "weight allocation budget");
	require(csr->w.empty() && !csr->initialized_w.load(), "weight budget refusal allocated storage");
	state.QueryEnd();
	refuses<ConstraintException>([&] { state.GetCSR(2); }, "failed weight allocation owner must be cleaned");
	// A separate complete owner isolates finalization's resident-plus-scratch check.
	auto complete = graph(state, 3, {n, 0});
	for (int64_t i = n - 1; i >= 0; --i)
		CSR::BeginEdgeWrite(complete, 0, CSRWeightMode::Unweighted).Commit(1, i);
	refuses<OutOfMemoryException>([&] { complete->FinalizeForRead(*connection.context); },
	                              "finalization resident and scratch budget");
	require(complete->edge_ids.front() == n - 1 && complete->edge_ids.back() == 0, "budget refusal mutated edge order");
	auto restored = connection.Query("SET memory_limit='256MB'");
	require(!restored->HasError(), "restore private budget");
	complete->FinalizeForRead(*connection.context);
	for (int64_t i = 0; i < n; ++i)
		require(complete->edge_ids[i] == i && complete->e[i] == 1, "budget positive control changed pairs");
}
int main() {
	DuckDB database(nullptr);
	Connection connection(database);
	auto settings = connection.Query("SET threads=1");
	require(!settings->HasError(), "native contract context setup failed");
	struct Case {
		const char *name;
		void (*run)(Connection &);
	};
	const Case cases[] = {
	    {"unweighted", unweighted},
	    {"int64", integer_weights},
	    {"double_bits", double_weights},
	    {"empty_isolated_padding", empty_and_isolated},
	    {"concurrent_finalize", concurrent_finalize},
	    {"concurrent_producers", concurrent_producers},
	    {"retained_owners", retained_owners},
	    {"incomplete_writes", incomplete_writes},
	    {"abandoned_single_commit", abandoned_and_single_commit},
	    {"late_writes", late_writes},
	    {"overreservation_modes", overreservation_and_modes},
	    {"invalid_input_shape", invalid_inputs_and_shapes},
	    {"failure_cleanup", failure_cleanup},
	    {"replacement_cleanup", replacement_cleanup},
	    {"budget_refusals", budget_refusals},
	};
	size_t passed = 0;
	for (const auto &test : cases) {
		try {
			test.run(connection);
			++passed;
			std::cout << "CSR_CONTRACT_PASS " << test.name << std::endl;
		} catch (const std::exception &error) {
			std::cerr << "CSR_CONTRACT_FAIL " << test.name << " " << error.what() << std::endl;
		}
	}
	std::cout << "CSR_FINALIZATION_CONTRACTS executed=15 passed=" << passed << " failed=" << (15 - passed) << std::endl;
	return passed == 15 ? 0 : 1;
}
