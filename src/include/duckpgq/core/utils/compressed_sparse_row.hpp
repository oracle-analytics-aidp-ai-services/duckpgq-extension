//===----------------------------------------------------------------------===//
//                         DuckPGQ
//
// duckpgq/core/utils/compressed_sparse_row.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once
#include "duckdb/function/function.hpp"

#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"

#include "duckdb/parser/property_graph_table.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckpgq/common.hpp"
#include <memory>
#include <mutex>

namespace duckdb {

enum class CSRWeightMode { Unweighted, Int64, Double };

class CSR {
public:
	CSR() = default;
	~CSR() {
		delete[] v;
	}

	// A reservation owns the CSR until its complete tuple is published. Abandoning
	// a reservation makes the construction unreadable; it never counts as a write.
	class EdgeWrite {
	public:
		EdgeWrite(EdgeWrite &&other) noexcept;
		EdgeWrite(const EdgeWrite &) = delete;
		EdgeWrite &operator=(const EdgeWrite &) = delete;
		~EdgeWrite();
		void Commit(int64_t dst, int64_t edge_id);
		void Commit(int64_t dst, int64_t edge_id, int64_t weight);
		void Commit(int64_t dst, int64_t edge_id, double weight);

	private:
		friend class CSR;
		EdgeWrite(shared_ptr<CSR> owner, size_t position, CSRWeightMode mode);
		void CommitBits(int64_t dst, int64_t edge_id, uint64_t bits, CSRWeightMode mode);
		shared_ptr<CSR> owner;
		size_t position;
		CSRWeightMode mode;
	};

	void SetVertexDegree(int64_t src, int64_t degree);
	static EdgeWrite BeginEdgeWrite(shared_ptr<CSR> owner, int64_t src, CSRWeightMode mode);
	// Call after the reader's existing initialization and working-budget checks.
	// This rejects incomplete producers, then orders each complete adjacency once.
	void FinalizeForRead(ClientContext &context);

	atomic<int64_t> *v {};

	vector<int64_t> e;
	vector<int64_t> edge_ids;

	vector<int64_t> w;
	vector<double> w_double;

	// These flags describe allocation only. FinalizeForRead requires publication
	// of every reserved neighbor/edge-ID/weight tuple before computation can read it.
	atomic<bool> initialized_v {false};
	atomic<bool> initialized_e {false};
	atomic<bool> initialized_w {false};

	size_t vsize {};

	string ToString() const;

private:
	friend class DuckPGQState;
	enum class Phase { Vertices, Edges, Finalized };
	std::mutex construction_lock;
	std::atomic<Phase> phase {Phase::Vertices};
	std::atomic<bool> write_failed {false};
	std::atomic<size_t> completed_writes {0};
	std::unique_ptr<std::atomic<int64_t>[]> write_cursors;
	size_t vertex_count = 0;
	size_t expected_writes = 0;
	std::atomic<bool> mode_set {false};
	CSRWeightMode weight_mode = CSRWeightMode::Unweighted;
	// Read only while construction_lock is held.
	idx_t ResidentBytes() const;
};

struct CSRFunctionData : FunctionData {
	CSRFunctionData(ClientContext &context, int32_t id, const LogicalType &weight_type);
	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other_p) const override;
	static unique_ptr<FunctionData> CSRVertexBind(ClientContext &context, ScalarFunction &bound_function,
	                                              vector<unique_ptr<Expression>> &arguments);
	static unique_ptr<FunctionData> CSREdgeBind(ClientContext &context, ScalarFunction &bound_function,
	                                            vector<unique_ptr<Expression>> &arguments);
	static unique_ptr<FunctionData> CSRBind(ClientContext &context, ScalarFunction &bound_function,
	                                        vector<unique_ptr<Expression>> &arguments);

	ClientContext &context;
	const int32_t id;
	const LogicalType weight_type;
};

// CSR BindReplace functions
unique_ptr<CommonTableExpressionInfo> CreateUndirectedCSRCTE(const shared_ptr<PropertyGraphTable> &edge_table,
                                                             const unique_ptr<SelectNode> &select_node);
unique_ptr<CommonTableExpressionInfo> CreateDirectedCSRCTE(const shared_ptr<PropertyGraphTable> &edge_table,
                                                           const string &prev_binding, const string &edge_binding,
                                                           const string &next_binding);

// Helper functions
unique_ptr<CommonTableExpressionInfo> MakeEdgesCTE(const shared_ptr<PropertyGraphTable> &edge_table);
unique_ptr<SubqueryExpression> CreateDirectedCSRVertexSubquery(const shared_ptr<PropertyGraphTable> &edge_table,
                                                               const string &binding);
unique_ptr<SubqueryExpression> CreateUndirectedCSRVertexSubquery(const shared_ptr<PropertyGraphTable> &edge_table,
                                                                 const string &binding);
unique_ptr<SelectNode> CreateOuterSelectEdgesNode();
unique_ptr<SelectNode> CreateOuterSelectNode(unique_ptr<FunctionExpression> create_csr_edge_function);
unique_ptr<JoinRef> GetJoinRef(const shared_ptr<PropertyGraphTable> &edge_table, const string &edge_binding,
                               const string &prev_binding, const string &next_binding);
unique_ptr<SubqueryExpression> GetCountTable(const shared_ptr<PropertyGraphTable> &table, const string &table_alias,
                                             const string &primary_key);
void SetupSelectNode(unique_ptr<SelectNode> &select_node, const shared_ptr<PropertyGraphTable> &edge_table,
                     bool reverse = false);
unique_ptr<SubqueryRef> CreateCountCTESubquery();
unique_ptr<SubqueryExpression> GetCountUndirectedEdgeTable();
unique_ptr<SubqueryExpression> GetCountEdgeTable(const shared_ptr<PropertyGraphTable> &edge_table);

} // namespace duckdb
