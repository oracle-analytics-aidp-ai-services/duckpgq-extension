#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckpgq/common.hpp"
#include "duckpgq/core/functions/function_data/hits_function_data.hpp"
#include <duckpgq/core/functions/scalar.hpp>
#include <duckpgq/core/utils/duckpgq_utils.hpp>

#include <cmath>

namespace duckdb {

// Shared HITS computation. Runs the full algorithm once and caches both the
// authority and hub vectors inside the HitsFunctionData bind data.
static void ComputeHits(HitsFunctionData &info) {
	auto duckpgq_state = GetDuckPGQState(info.context);

	// Locate the CSR representation of the graph
	auto csr = duckpgq_state->GetCSR(info.csr_id, "CSR not found. Is the graph populated?");

	if (!(csr->initialized_v && csr->initialized_e)) {
		throw ConstraintException("Need to initialize CSR before running HITS.");
	}

	auto *v = reinterpret_cast<int64_t *>(csr->v);
	vector<int64_t> &e = csr->e;
	size_t v_size = csr->vsize;

	CheckAlgorithmMemoryBudget(info.context, (idx_t)v_size * sizeof(double) * 4, "hits");
	duckpgq_state->FinalizeCSRForRead(info.csr_id, csr, info.context);

	// State initialization / computation (only once)
	if (info.state_initialized) {
		return;
	}
	std::lock_guard<std::mutex> guard(info.state_lock); // Thread safety
	if (info.state_initialized) {
		return; // Another thread finished while we waited for the lock
	}

	info.authority.assign(v_size, 1.0);
	info.hub.assign(v_size, 1.0);

	vector<double> new_auth(v_size, 0.0);
	vector<double> new_hub(v_size, 0.0);

	const int64_t max_iterations = 100;
	const double convergence_threshold = 1e-8;

	for (int64_t iter = 0; iter < max_iterations; iter++) {
		std::fill(new_auth.begin(), new_auth.end(), 0.0);
		std::fill(new_hub.begin(), new_hub.end(), 0.0);

		// Authority update: auth[w] += hub[u] for each edge u->w
		for (size_t u = 0; u < v_size; u++) {
			int64_t start_edge = v[u];
			int64_t end_edge = (u + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
			for (int64_t j = start_edge; j < end_edge; j++) {
				int64_t w = e[j];
				new_auth[w] += info.hub[u];
			}
		}

		// Hub update: hub[u] += auth[w] for each edge u->w (using new authorities)
		for (size_t u = 0; u < v_size; u++) {
			int64_t start_edge = v[u];
			int64_t end_edge = (u + 1 < v_size) ? v[u + 1] : static_cast<int64_t>(e.size());
			for (int64_t j = start_edge; j < end_edge; j++) {
				int64_t w = e[j];
				new_hub[u] += new_auth[w];
			}
		}

		// Normalize each vector (L2)
		double auth_norm = 0.0;
		double hub_norm = 0.0;
		for (size_t i = 0; i < v_size; i++) {
			auth_norm += new_auth[i] * new_auth[i];
			hub_norm += new_hub[i] * new_hub[i];
		}
		auth_norm = std::sqrt(auth_norm);
		hub_norm = std::sqrt(hub_norm);
		if (auth_norm > 0.0) {
			for (size_t i = 0; i < v_size; i++) {
				new_auth[i] /= auth_norm;
			}
		}
		if (hub_norm > 0.0) {
			for (size_t i = 0; i < v_size; i++) {
				new_hub[i] /= hub_norm;
			}
		}

		// Convergence check (max absolute delta across both vectors)
		double max_delta = 0.0;
		for (size_t i = 0; i < v_size; i++) {
			max_delta = std::max(max_delta, std::abs(new_auth[i] - info.authority[i]));
			max_delta = std::max(max_delta, std::abs(new_hub[i] - info.hub[i]));
		}

		info.authority = new_auth;
		info.hub = new_hub;

		if (max_delta < convergence_threshold) {
			break;
		}
	}

	info.state_initialized = true;
}

static void EmitResult(DataChunk &args, Vector &result, const vector<double> &values, size_t v_size) {
	// Get the source vector for the current DataChunk
	auto &src = args.data[1];
	UnifiedVectorFormat vdata_src;
	src.ToUnifiedFormat(args.size(), vdata_src);
	auto src_data = reinterpret_cast<const int64_t *>(vdata_src.data);

	ValidityMask &result_validity = FlatVector::Validity(result);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto result_data = FlatVector::GetData<double_t>(result);

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
		result_data[i] = values[node_id];
	}
}

static void HitsAuthorityFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<HitsFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	ComputeHits(info);

	EmitResult(args, result, info.authority, info.authority.size());

	duckpgq_state->ScheduleDelete(info.csr_id);
}

static void HitsHubFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &func_expr = state.expr.Cast<BoundFunctionExpression>();
	auto &info = func_expr.bind_info->Cast<HitsFunctionData>();
	auto duckpgq_state = GetDuckPGQState(info.context);

	ComputeHits(info);

	EmitResult(args, result, info.hub, info.hub.size());

	duckpgq_state->ScheduleDelete(info.csr_id);
}

//------------------------------------------------------------------------------
// Register functions
//------------------------------------------------------------------------------
void CoreScalarFunctions::RegisterHitsAuthorityScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("hits_authority", {LogicalType::INTEGER, LogicalType::BIGINT},
	                                       LogicalType::DOUBLE, HitsAuthorityFunction, HitsFunctionData::HitsBind));
}

void CoreScalarFunctions::RegisterHitsHubScalarFunction(ExtensionLoader &loader) {
	loader.RegisterFunction(ScalarFunction("hits_hub", {LogicalType::INTEGER, LogicalType::BIGINT}, LogicalType::DOUBLE,
	                                       HitsHubFunction, HitsFunctionData::HitsBind));
}

} // namespace duckdb
