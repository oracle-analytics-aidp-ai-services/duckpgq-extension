// Standalone contract regression: compile with src/duckpgq_state.cpp and the
// matching DuckDB library. No SQL connector, fake registry, or timing sleeps.
#include "duckpgq_state.hpp"
#include <atomic>
#include <iostream>
#include <thread>
#include <type_traits>
#include <vector>

using namespace duckdb;
static_assert(std::is_same<decltype(std::declval<DuckPGQState &>().GetCSR(0)), shared_ptr<CSR>>::value,
              "CSR acquisition must retain ownership beyond registry deletion");

int main() {
	DuckPGQState state;
	auto retained = state.InitializeVertex(0, 16);
	retained->v[0] = 73;
	if (!state.DeleteCSR(0) || retained->v[0] != 73)
		return 1;
	auto replacement = state.InitializeVertex(0, 16);
	if (replacement.get() == retained.get() || replacement->v[0] != 0)
		return 2;
	state.ScheduleDelete(0);
	state.QueryEnd();
	if (replacement->vsize != 18 || retained->v[0] != 73)
		return 3;
	bool missing = false;
	try {
		state.GetCSR(0);
	} catch (const ConstraintException &) {
		missing = true;
	}
	if (!missing)
		return 4;

	auto integer_weights = state.InitializeVertex(-1, 2);
	integer_weights->v[2] = 1;
	state.InitializeEdges(-1, 2, 1);
	state.InitializeWeights(-1, 1, PhysicalType::INT64)->w[0] = 7;
	if (state.InitializeWeights(-1, 1, PhysicalType::INT64)->w[0] != 7)
		return 6;
	auto double_weights = state.InitializeVertex(-2, 2);
	double_weights->v[2] = 1;
	state.InitializeEdges(-2, 2, 1);
	state.InitializeWeights(-2, 1, PhysicalType::DOUBLE)->w_double[0] = 7.5;
	if (state.InitializeWeights(-2, 1, PhysicalType::DOUBLE)->w_double[0] != 7.5)
		return 7;

	// Simultaneous publication of the same and different IDs: every acquired
	// object is initialized and every caller of one ID receives the same object.
	std::atomic<bool> start {false};
	std::atomic<bool> failed {false};
	std::vector<std::thread> workers;
	for (int worker = 0; worker < 8; ++worker) {
		workers.emplace_back([&] {
			while (!start.load())
				std::this_thread::yield();
			for (int round = 0; round < 64; ++round) {
				for (int id = 1; id <= 64; ++id) {
					auto created = state.InitializeVertex(id, 16);
					auto acquired = state.GetCSR(id);
					if (created.get() != acquired.get() || !acquired->initialized_v || acquired->vsize != 18)
						failed = true;
					state.ScheduleDelete(id);
				}
			}
		});
	}
	start = true;
	for (auto &worker : workers)
		worker.join();
	state.QueryEnd();
	if (failed)
		return 5;
	std::cout << "CSR ownership and parallel publication passed\n";
	return 0;
}
