//===----------------------------------------------------------------------===//
//                         DuckPGQ
//
// duckpgq/core/functions/table/single_source_shortest_path.hpp
//
//
//===----------------------------------------------------------------------===//

#include "duckpgq/common.hpp"

namespace duckdb {

class SingleSourceShortestPathFunction : public TableFunction {
public:
	SingleSourceShortestPathFunction() {
		name = "single_source_shortest_path";
		// single_source_shortest_path(vertex_table, vertex_id_col, edge_table, src_col, dst_col, source_vertex)
		arguments = {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
		             LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT};
		bind_replace = SingleSourceShortestPathBindReplace;
	}

	static unique_ptr<TableRef> SingleSourceShortestPathBindReplace(ClientContext &context,
	                                                                TableFunctionBindInput &input);
};

struct SingleSourceShortestPathData : TableFunctionData {
	static unique_ptr<FunctionData> SingleSourceShortestPathBind(ClientContext &context, TableFunctionBindInput &input,
	                                                             vector<LogicalType> &return_types,
	                                                             vector<string> &names) {
		auto result = make_uniq<SingleSourceShortestPathData>();
		result->pg_name = StringValue::Get(input.inputs[0]);
		result->node_table = StringValue::Get(input.inputs[1]);
		result->edge_table = StringValue::Get(input.inputs[2]);
		// inputs[3] (source) is not needed for the schema.
		return_types.emplace_back(LogicalType::BIGINT);
		return_types.emplace_back(LogicalType::BIGINT);
		names.emplace_back("rowid");
		names.emplace_back("distance");
		return std::move(result);
	}

	string pg_name;
	string node_table;
	string edge_table;
};

struct SingleSourceShortestPathScanState : GlobalTableFunctionState {
	static unique_ptr<GlobalTableFunctionState> Init(ClientContext &context, TableFunctionInitInput &input) {
		auto result = make_uniq<SingleSourceShortestPathScanState>();
		return std::move(result);
	}

	bool finished = false;
};

} // namespace duckdb
