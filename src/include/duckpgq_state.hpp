#pragma once

#include "duckpgq/common.hpp"
#include "duckpgq/core/utils/compressed_sparse_row.hpp"

#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {

// Stock-DuckDB build: DuckPGQState carries only the CSR cache used by the graph
// algorithms / path-finding functions. The property-graph registry and parser-
// extension state were dropped together with the SQL/PGQ grammar front-end —
// graphs are described by table-function arguments, not registered objects.
class DuckPGQState : public ClientContextState {
public:
	explicit DuckPGQState() {};

	void QueryEnd() override;
	shared_ptr<CSR> GetCSR(int32_t id, const char *missing_message = nullptr);
	shared_ptr<CSR> InitializeVertex(int32_t id, int64_t vertex_count);
	shared_ptr<CSR> InitializeEdges(int32_t id, int64_t vertex_count, int64_t edge_count);
	shared_ptr<CSR> InitializeWeights(int32_t id, int64_t edge_count, PhysicalType weight_type);
	bool DeleteCSR(int32_t id);
	void ScheduleDelete(int32_t id);


private:
	shared_ptr<CSR> GetCSRLocked(int32_t id, const char *missing_message = nullptr);
	//! CSR data structures built for graph-algorithm / path-finding queries
	std::unordered_map<int32_t, shared_ptr<CSR>> csr_list;
	std::mutex csr_lock;
	std::unordered_set<int32_t> csr_to_delete;
};

} // namespace duckdb
