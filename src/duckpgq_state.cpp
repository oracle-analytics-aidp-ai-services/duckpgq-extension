#include "duckpgq_state.hpp"
#include <limits>

namespace duckdb {

void DuckPGQState::QueryEnd() {
	std::lock_guard<std::mutex> guard(csr_lock);
	for (const auto &csr_id : csr_to_delete) {
		csr_list.erase(csr_id);
	}
	csr_to_delete.clear();
}

shared_ptr<CSR> DuckPGQState::GetCSRLocked(int32_t id, const char *missing_message) {
	auto csr_entry = csr_list.find(id);
	if (csr_entry == csr_list.end()) {
		if (missing_message) {
			throw ConstraintException("%s", missing_message);
		}
		throw ConstraintException("CSR not found with ID %d", id);
	}
	return csr_entry->second;
}

shared_ptr<CSR> DuckPGQState::GetCSR(int32_t id, const char *missing_message) {
	std::lock_guard<std::mutex> guard(csr_lock);
	return GetCSRLocked(id, missing_message);
}

bool DuckPGQState::DeleteCSR(int32_t id) {
	std::lock_guard<std::mutex> guard(csr_lock);
	csr_to_delete.erase(id);
	return csr_list.erase(id) != 0;
}

void DuckPGQState::ScheduleDelete(int32_t id) {
	std::lock_guard<std::mutex> guard(csr_lock);
	csr_to_delete.insert(id);
}

shared_ptr<CSR> DuckPGQState::InitializeVertex(int32_t id, int64_t vertex_count) {
	std::lock_guard<std::mutex> guard(csr_lock);
	auto existing = csr_list.find(id);
	if (existing != csr_list.end()) return existing->second;
	if (vertex_count < 0 || static_cast<uint64_t>(vertex_count) >
	    (std::numeric_limits<size_t>::max() / sizeof(std::atomic<int64_t>)) - 2) {
		throw InvalidInputException("Invalid CSR vertex count");
	}
	auto csr = make_shared_ptr<CSR>();
	csr->vsize = static_cast<size_t>(vertex_count) + 2;
	try {
		csr->v = new std::atomic<int64_t>[csr->vsize];
	} catch (std::bad_alloc const &) {
		throw Exception(ExceptionType::INTERNAL, "Unable to initialize vector of size for csr vertex table representation");
	}
	for (size_t i = 0; i < csr->vsize; ++i) csr->v[i] = 0;
	csr->initialized_v = true;
	// Publish only a complete non-null owner; readers acquire the same mutex.
	csr_list.emplace(id, csr);
	return csr;
}

shared_ptr<CSR> DuckPGQState::InitializeEdges(int32_t id, int64_t vertex_count, int64_t edge_count) {
	std::lock_guard<std::mutex> guard(csr_lock);
	auto csr = GetCSRLocked(id);
	if (csr->initialized_e) return csr;
	if (vertex_count < 0 || static_cast<uint64_t>(vertex_count) != csr->vsize - 2 || edge_count < 0) {
		throw InvalidInputException("Invalid CSR edge dimensions");
	}
	try {
		csr->e.resize(edge_count, 0);
		csr->edge_ids.resize(edge_count, 0);
	} catch (std::bad_alloc const &) {
		throw Exception(ExceptionType::INTERNAL, "Unable to initialize vector of size for csr edge table representation");
	}
	for (size_t i = 1; i < csr->vsize; ++i) csr->v[i] += csr->v[i - 1];
	csr->initialized_e = true;
	return csr;
}

shared_ptr<CSR> DuckPGQState::InitializeWeights(int32_t id, int64_t edge_count, PhysicalType weight_type) {
	std::lock_guard<std::mutex> guard(csr_lock);
	auto csr = GetCSRLocked(id);
	if (csr->initialized_w) return csr;
	if (edge_count < 0 || static_cast<uint64_t>(edge_count) != csr->e.size()) {
		throw InvalidInputException("Invalid CSR weight count");
	}
	try {
		if (weight_type == PhysicalType::INT64) csr->w.resize(edge_count, 0);
		else if (weight_type == PhysicalType::DOUBLE) csr->w_double.resize(edge_count, 0);
		else throw NotImplementedException("Unrecognized weight type detected.");
	} catch (std::bad_alloc const &) {
		throw Exception(ExceptionType::INTERNAL, "Unable to initialize vector of size for csr weight table representation");
	}
	csr->initialized_w = true;
	return csr;
}

} // namespace duckdb
