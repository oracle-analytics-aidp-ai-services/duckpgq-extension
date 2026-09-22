#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/label_propagation_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/functions/table/label_propagation.hpp>
#include <duckpgq/core/utils/duckpgq_bitmap.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>
#include <unordered_map>

namespace duckdb {

static void LabelPropagationFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<LabelPropagationFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before running label propagation.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;
	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(int64_t) * 2, "label_propagation");
	duckpgq_state->FinalizeCSRForRead(info.csr_id, csr, info.context);

	// State initialization and computation (only once)
	if (!info.state_initialized) {
		std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
		if (!info.state_initialized) {
			// Initialize each vertex with its own label
			info.label.resize(v_size);
			for (size_t i = 0; i < v_size; i++) {
				info.label[i] = static_cast<int64_t>(i);
			}

			vector<int64_t> newlabel(v_size, 0);
			const int64_t max_iterations = 20; // heuristic: cap iterations (cf. Neo4j GDS default ~10) to bound runtime
			                                   // on large/near-regular graphs
			for (int64_t iter = 0; iter < max_iterations; iter++) {
				bool changed = false;
				std::unordered_map<int64_t, int64_t> counts; // reused across vertices to avoid reallocation
				for (size_t u = 0; u + 2 < v_size; u++) {
					auto start_edge = v[u];
					auto end_edge = (u + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
					if (end_edge <= start_edge) {
						// No neighbors: keep current label
						newlabel[u] = info.label[u];
						continue;
					}

					// Count label frequencies among neighbors using the
					// previous iteration's labels (synchronous update).
					counts.clear();
					for (int64_t j = start_edge; j < end_edge; j++) {
						int64_t neighbor = e[j];
						counts[info.label[neighbor]]++;
					}

					// Pick the most frequent label, breaking ties by the
					// smallest label value for determinism.
					int64_t best_label = info.label[u];
					int64_t best_count = -1;
					for (auto &entry : counts) {
						if (entry.second > best_count || (entry.second == best_count && entry.first < best_label)) {
							best_count = entry.second;
							best_label = entry.first;
						}
					}
					newlabel[u] = best_label;
					if (newlabel[u] != info.label[u]) {
						changed = true;
					}
				}
				info.label.swap(newlabel);
				if (!changed) {
					break;
				}
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

	// Output the label corresponding to each source ID in the DataChunk
	for (idx_t i = 0; i < args.size(); i++) {
		auto id_pos = vdata_src.sel->get_index(i);
		if (!vdata_src.validity.RowIsValid(id_pos)) {
			result_validity.SetInvalid(i);
			continue;
		}
		auto node_id = src_data[id_pos];
		if (node_id < 0 || static_cast<size_t>(node_id) >= v_size) {
			result_validity.SetInvalid(i);
			continue;
		}
		result_data[i] = info.label[node_id];
	}

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterLabelPropagationScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("label_propagation", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::BIGINT, LabelPropagationFunction,
	                                       LabelPropagationFunctionData::LabelPropagationBind));
}

} // namespace duckdb
