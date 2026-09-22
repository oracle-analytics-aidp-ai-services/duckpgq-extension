#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/strongly_connected_component_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>
#include <algorithm>
#include <vector>

namespace duckdb {

static void StronglyConnectedComponentFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<StronglyConnectedComponentFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before computing strongly connected components.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t) * 7, "strongly_connected_component");
	duckpgq_state->FinalizeCSRForRead(info.csr_id, csr, info.context);

	// Compute the strongly connected components once and cache the result.
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
		if (!info.state_initialized) {
			// Iterative (explicit-stack) Tarjan's SCC algorithm to avoid stack
			// overflow on large graphs.
			const int64_t UNVISITED = -1;
			vector<int64_t> disc(v_size, UNVISITED); // Discovery index per vertex
			vector<int64_t> low(v_size, 0);          // Low-link value per vertex
			vector<bool> on_stack(v_size, false);    // Whether vertex is on the SCC stack
			vector<int64_t> scc_stack;               // Tarjan's component stack
			vector<int64_t> component(v_size, UNVISITED);
			int64_t index_counter = 0;

			// Explicit DFS stack: each frame tracks the vertex and the index of
			// the next out-neighbor to process.
			struct Frame {
				int64_t node;
				int64_t edge_idx;
			};
			vector<Frame> call_stack;

			for (size_t root = 0; root < v_size; root++) {
				if (disc[root] != UNVISITED) {
					continue;
				}
				call_stack.push_back({static_cast<int64_t>(root), v[root]});
				disc[root] = index_counter;
				low[root] = index_counter;
				index_counter++;
				scc_stack.push_back(static_cast<int64_t>(root));
				on_stack[root] = true;

				while (!call_stack.empty()) {
					auto &frame = call_stack.back();
					int64_t u = frame.node;
					int64_t end_edge =
					    (static_cast<size_t>(u) + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());

					if (frame.edge_idx < end_edge) {
						int64_t w = e[frame.edge_idx];
						frame.edge_idx++;
						if (disc[w] == UNVISITED) {
							// Descend into unvisited neighbor.
							disc[w] = index_counter;
							low[w] = index_counter;
							index_counter++;
							scc_stack.push_back(w);
							on_stack[w] = true;
							call_stack.push_back({w, v[w]});
						} else if (on_stack[w]) {
							low[u] = std::min(low[u], disc[w]);
						}
					} else {
						// Done exploring u: if it is a root of an SCC, pop it.
						if (low[u] == disc[u]) {
							while (true) {
								int64_t node = scc_stack.back();
								scc_stack.pop_back();
								on_stack[node] = false;
								component[node] = u; // Temporary representative
								if (node == u) {
									break;
								}
							}
						}
						call_stack.pop_back();
						if (!call_stack.empty()) {
							int64_t parent = call_stack.back().node;
							low[parent] = std::min(low[parent], low[u]);
						}
					}
				}
			}

			// Relabel components so each vertex's component id is the smallest
			// vertex id contained in that component (deterministic output).
			vector<int64_t> min_label(v_size, UNVISITED);
			for (size_t i = 0; i < v_size; i++) {
				int64_t rep = component[i];
				if (min_label[rep] == UNVISITED || static_cast<int64_t>(i) < min_label[rep]) {
					min_label[rep] = static_cast<int64_t>(i);
				}
			}
			info.component.resize(v_size);
			for (size_t i = 0; i < v_size; i++) {
				info.component[i] = min_label[component[i]];
			}
			info.state_initialized = true;
		}
	}

	// Get the source vector for the current DataChunk
	auto &src = args.data[1];
	UnifiedVectorFormat vdata_src;
	src.ToUnifiedFormat(args.size(), vdata_src);
	auto src_data = reinterpret_cast<const int64_t *>(vdata_src.data);

	// Create result vector
	ValidityMask &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<int64_t>(result);

	// Output the component id corresponding to each source rowid in the DataChunk
	for (idx_t i = 0; i < args.size(); i++) {
		auto id_pos = vdata_src.sel->get_index(i);
		if (!vdata_src.validity.RowIsValid(id_pos)) {
			result_validity.SetInvalid(i);
			continue; // Skip invalid rows
		}
		auto node_id = src_data[id_pos];
		if (node_id < 0 || static_cast<size_t>(node_id) >= v_size) {
			result_validity.SetInvalid(i);
			continue;
		}
		result_data[i] = info.component[node_id];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterStronglyConnectedComponentScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("strongly_connected_component", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::BIGINT, StronglyConnectedComponentFunction,
	                                       StronglyConnectedComponentFunctionData::StronglyConnectedComponentBind));
}

} // namespace duckdb
