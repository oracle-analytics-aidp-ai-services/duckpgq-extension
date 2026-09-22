#include "duckpgq_state.hpp"
#include "duckpgq/core/utils/duckpgq_utils.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

namespace duckdb {

namespace {
idx_t CheckedBytes(size_t count, size_t width) {
	if (count > std::numeric_limits<idx_t>::max() / width) {
		throw OutOfMemoryException("CSR memory estimate overflow");
	}
	return static_cast<idx_t>(count) * width;
}

idx_t CheckedAdd(idx_t left, idx_t right) {
	if (right > std::numeric_limits<idx_t>::max() - left) {
		throw OutOfMemoryException("CSR memory estimate overflow");
	}
	return left + right;
}

struct EdgeRecord {
	int64_t neighbor;
	int64_t edge_id;
	uint64_t weight_bits;
};
static_assert(sizeof(EdgeRecord) == 24, "CSR ordering scratch accounting must match its record");
} // namespace

idx_t CSR::ResidentBytes() const {
	auto bytes = CheckedAdd(sizeof(CSR), CheckedBytes(vertex_count + 2, sizeof(std::atomic<int64_t>)));
	bytes = CheckedAdd(bytes, CheckedBytes(e.capacity(), sizeof(int64_t)));
	bytes = CheckedAdd(bytes, CheckedBytes(edge_ids.capacity(), sizeof(int64_t)));
	bytes = CheckedAdd(bytes, CheckedBytes(w.capacity(), sizeof(int64_t)));
	bytes = CheckedAdd(bytes, CheckedBytes(w_double.capacity(), sizeof(double)));
	if (write_cursors) {
		bytes = CheckedAdd(bytes, CheckedBytes(vertex_count, sizeof(std::atomic<int64_t>)));
	}
	return bytes;
}

void CSR::SetVertexDegree(int64_t src, int64_t degree) {
	std::lock_guard<std::mutex> guard(construction_lock);
	if (phase.load(std::memory_order_acquire) != Phase::Vertices) {
		throw ConstraintException("CSR vertex construction is sealed");
	}
	if (!initialized_v || vsize != vertex_count + 2 || src < 0 || static_cast<uint64_t>(src) >= vertex_count ||
	    degree < 0) {
		throw InvalidInputException("Invalid CSR vertex degree");
	}
	v[src + 2] = degree;
}

CSR::EdgeWrite::EdgeWrite(shared_ptr<CSR> owner_p, size_t position_p, CSRWeightMode mode_p)
    : owner(std::move(owner_p)), position(position_p), mode(mode_p) {
}

CSR::EdgeWrite::EdgeWrite(EdgeWrite &&other) noexcept
    : owner(std::move(other.owner)), position(other.position), mode(other.mode) {
}

CSR::EdgeWrite::~EdgeWrite() {
	if (owner) {
		owner->write_failed.store(true, std::memory_order_release);
	}
}

CSR::EdgeWrite CSR::BeginEdgeWrite(shared_ptr<CSR> owner, int64_t src, CSRWeightMode mode) {
	if (!owner || owner->vsize != owner->vertex_count + 2 || src < 0 ||
	    static_cast<uint64_t>(src) >= owner->vertex_count ||
	    (mode != CSRWeightMode::Unweighted && mode != CSRWeightMode::Int64 && mode != CSRWeightMode::Double)) {
		throw InvalidInputException("Invalid CSR edge reservation");
	}
	if (owner->phase.load(std::memory_order_acquire) != Phase::Edges ||
	    owner->write_failed.load(std::memory_order_acquire)) {
		throw ConstraintException("CSR edge construction is not writable");
	}
	if (!owner->mode_set.load(std::memory_order_acquire)) {
		std::lock_guard<std::mutex> guard(owner->construction_lock);
		if (owner->phase.load(std::memory_order_acquire) != Phase::Edges) {
			throw ConstraintException("CSR edge construction is sealed");
		}
		if (!owner->mode_set.load(std::memory_order_relaxed)) {
			if (mode != CSRWeightMode::Unweighted) {
				throw ConstraintException("CSR weights must be initialized before writing edges");
			}
			owner->weight_mode = mode;
			owner->mode_set.store(true, std::memory_order_release);
		}
	}
	if (owner->weight_mode != mode) {
		throw ConstraintException("Mixed CSR edge weight modes");
	}
	const auto end = owner->v[src + 1].load(std::memory_order_relaxed);
	auto &cursor = owner->write_cursors[src];
	auto position = cursor.load(std::memory_order_relaxed);
	do {
		if (position >= end) {
			throw ConstraintException("CSR vertex edge reservation exceeded its degree");
		}
	} while (!cursor.compare_exchange_weak(position, position + 1, std::memory_order_relaxed));
	return EdgeWrite(std::move(owner), static_cast<size_t>(position), mode);
}

void CSR::EdgeWrite::CommitBits(int64_t dst, int64_t edge_id, uint64_t bits, CSRWeightMode supplied_mode) {
	if (!owner) {
		throw ConstraintException("CSR edge reservation already completed");
	}
	if (mode != supplied_mode || owner->write_failed.load(std::memory_order_acquire)) {
		owner->write_failed.store(true, std::memory_order_release);
		throw ConstraintException("CSR edge reservation cannot complete");
	}
	if (dst < 0 || static_cast<uint64_t>(dst) >= owner->vertex_count) {
		owner->write_failed.store(true, std::memory_order_release);
		throw InvalidInputException("Invalid CSR edge destination");
	}
	if (position >= owner->e.size() || position >= owner->edge_ids.size() ||
	    (mode == CSRWeightMode::Int64 && position >= owner->w.size()) ||
	    (mode == CSRWeightMode::Double && position >= owner->w_double.size())) {
		owner->write_failed.store(true, std::memory_order_release);
		throw ConstraintException("Invalid CSR edge storage");
	}
	owner->e[position] = dst;
	owner->edge_ids[position] = edge_id;
	if (mode == CSRWeightMode::Int64) {
		std::memcpy(&owner->w[position], &bits, sizeof(bits));
	}
	if (mode == CSRWeightMode::Double) {
		std::memcpy(&owner->w_double[position], &bits, sizeof(bits));
	}
	// The final acquire observes every complete paired write in this RMW chain.
	owner->completed_writes.fetch_add(1, std::memory_order_acq_rel);
	owner.reset();
}

void CSR::EdgeWrite::Commit(int64_t dst, int64_t edge_id) {
	CommitBits(dst, edge_id, 0, CSRWeightMode::Unweighted);
}

void CSR::EdgeWrite::Commit(int64_t dst, int64_t edge_id, int64_t weight) {
	uint64_t bits;
	std::memcpy(&bits, &weight, sizeof(bits));
	CommitBits(dst, edge_id, bits, CSRWeightMode::Int64);
}

void CSR::EdgeWrite::Commit(int64_t dst, int64_t edge_id, double weight) {
	static_assert(sizeof(double) == sizeof(uint64_t), "CSR double weights require 64-bit storage");
	uint64_t bits;
	std::memcpy(&bits, &weight, sizeof(bits));
	CommitBits(dst, edge_id, bits, CSRWeightMode::Double);
}

void CSR::FinalizeForRead(ClientContext &context) {
	std::lock_guard<std::mutex> guard(construction_lock);
	if (phase.load(std::memory_order_acquire) == Phase::Finalized) {
		return;
	}
	if (phase.load(std::memory_order_relaxed) != Phase::Edges || write_failed.load(std::memory_order_acquire) ||
	    completed_writes.load(std::memory_order_acquire) != expected_writes) {
		throw ConstraintException("CSR paired edge construction is incomplete");
	}
	if (!initialized_v || !initialized_e || vsize != vertex_count + 2 || !v || e.size() != edge_ids.size() ||
	    expected_writes > e.size() || v[0] != 0 || v[vsize - 2] != static_cast<int64_t>(expected_writes)) {
		throw ConstraintException("Invalid completed CSR shape");
	}
	if ((weight_mode == CSRWeightMode::Unweighted && (!w.empty() || !w_double.empty() || initialized_w)) ||
	    (weight_mode == CSRWeightMode::Int64 && (!initialized_w || w.size() != e.size() || !w_double.empty())) ||
	    (weight_mode == CSRWeightMode::Double && (!initialized_w || w_double.size() != e.size() || !w.empty()))) {
		throw ConstraintException("Invalid completed CSR weight storage");
	}
	size_t max_degree = 0;
	for (size_t src = 0; src < vsize - 2; ++src) {
		const auto begin = v[src].load();
		const auto end = v[src + 1].load();
		if (begin < 0 || end < begin || static_cast<uint64_t>(end) > expected_writes ||
		    write_cursors[src].load(std::memory_order_acquire) != end) {
			throw ConstraintException("Invalid completed CSR offsets");
		}
		max_degree = std::max(max_degree, static_cast<size_t>(end - begin));
	}
	for (size_t index = 0; index < expected_writes; ++index) {
		if (e[index] < 0 || static_cast<uint64_t>(e[index]) >= vsize - 2) {
			throw ConstraintException("Invalid completed CSR neighbor");
		}
	}
	CheckAlgorithmMemoryBudget(context, CheckedAdd(ResidentBytes(), CheckedBytes(max_degree, sizeof(EdgeRecord))),
	                           "csr_adjacency_order");
	std::vector<EdgeRecord> scratch;
	try {
		scratch.resize(max_degree);
	} catch (const std::bad_alloc &) {
		throw OutOfMemoryException("Unable to allocate CSR adjacency ordering scratch");
	}
	for (size_t src = 0; src < vsize - 2; ++src) {
		const auto begin = static_cast<size_t>(v[src].load());
		const auto end = static_cast<size_t>(v[src + 1].load());
		const auto degree = end - begin;
		for (size_t index = 0; index < degree; ++index) {
			auto &record = scratch[index];
			record.neighbor = e[begin + index];
			record.edge_id = edge_ids[begin + index];
			record.weight_bits = 0;
			if (weight_mode == CSRWeightMode::Int64) {
				std::memcpy(&record.weight_bits, &w[begin + index], sizeof(uint64_t));
			}
			if (weight_mode == CSRWeightMode::Double) {
				std::memcpy(&record.weight_bits, &w_double[begin + index], sizeof(uint64_t));
			}
		}
		// Each degree is bounded by the allocated scratch vector size.
		std::sort(scratch.begin(), scratch.begin() + static_cast<std::vector<EdgeRecord>::difference_type>(degree),
		          [](const EdgeRecord &a, const EdgeRecord &b) {
			          if (a.neighbor != b.neighbor) {
				          return a.neighbor < b.neighbor;
			          }
			          if (a.edge_id != b.edge_id) {
				          return a.edge_id < b.edge_id;
			          }
			          return a.weight_bits < b.weight_bits;
		          });
		for (size_t index = 0; index < degree; ++index) {
			const auto &record = scratch[index];
			e[begin + index] = record.neighbor;
			edge_ids[begin + index] = record.edge_id;
			if (weight_mode == CSRWeightMode::Int64) {
				std::memcpy(&w[begin + index], &record.weight_bits, sizeof(uint64_t));
			}
			if (weight_mode == CSRWeightMode::Double) {
				std::memcpy(&w_double[begin + index], &record.weight_bits, sizeof(uint64_t));
			}
		}
	}
	phase.store(Phase::Finalized, std::memory_order_release);
}

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

void DuckPGQState::ScheduleDeleteIfOwned(int32_t id, const shared_ptr<CSR> &owner) {
	std::lock_guard<std::mutex> guard(csr_lock);
	auto current = csr_list.find(id);
	if (current != csr_list.end() && current->second.get() == owner.get()) {
		csr_to_delete.insert(id);
	}
}

void DuckPGQState::FinalizeCSRForRead(int32_t id, const shared_ptr<CSR> &owner, ClientContext &context) {
	try {
		if (!owner) {
			throw ConstraintException("CSR owner is missing");
		}
		owner->FinalizeForRead(context);
	} catch (...) {
		// QueryEnd removes only this registry entry; retained owners stay alive.
		ScheduleDeleteIfOwned(id, owner);
		throw;
	}
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
	if (existing != csr_list.end()) {
		return existing->second;
	}
	if (vertex_count < 0 ||
	    static_cast<uint64_t>(vertex_count) > (std::numeric_limits<size_t>::max() / sizeof(std::atomic<int64_t>)) - 2) {
		throw InvalidInputException("Invalid CSR vertex count");
	}
	auto csr = make_shared_ptr<CSR>();
	csr->vsize = static_cast<size_t>(vertex_count) + 2;
	csr->vertex_count = static_cast<size_t>(vertex_count);
	try {
		csr->v = new std::atomic<int64_t>[csr->vsize];
	} catch (std::bad_alloc const &) {
		throw Exception(ExceptionType::INTERNAL,
		                "Unable to initialize vector of size for csr vertex table representation");
	}
	for (size_t i = 0; i < csr->vsize; ++i) {
		csr->v[i] = 0;
	}
	csr->initialized_v = true;
	// Publish only a complete non-null owner; readers acquire the same mutex.
	csr_list.emplace(id, csr);
	return csr;
}

shared_ptr<CSR> DuckPGQState::InitializeEdges(int32_t id, int64_t vertex_count, int64_t edge_count,
                                              ClientContext *context) {
	auto csr = GetCSR(id);
	try {
		std::lock_guard<std::mutex> guard(csr->construction_lock);
		if (csr->phase.load() == CSR::Phase::Finalized || csr->write_failed.load()) {
			throw ConstraintException("CSR edge construction is sealed");
		}
		if (vertex_count < 0 || static_cast<uint64_t>(vertex_count) != csr->vertex_count ||
		    csr->vsize != csr->vertex_count + 2 || edge_count < 0) {
			throw InvalidInputException("Invalid CSR edge dimensions");
		}
		if (csr->initialized_e) {
			if (static_cast<uint64_t>(edge_count) != csr->e.size()) {
				throw InvalidInputException("Invalid CSR edge dimensions");
			}
			return csr;
		}
		int64_t expected = 0;
		for (size_t src = 0; src < static_cast<size_t>(vertex_count); ++src) {
			const auto degree = csr->v[src + 2].load();
			if (degree < 0 || degree > edge_count - expected) {
				throw InvalidInputException("Invalid CSR vertex degree sum");
			}
			expected += degree;
		}
		const auto cursor_bytes = CheckedBytes(static_cast<size_t>(vertex_count), sizeof(std::atomic<int64_t>));
		const auto edge_bytes = CheckedBytes(static_cast<size_t>(edge_count), sizeof(int64_t) * 2);
		if (context) {
			CheckAlgorithmMemoryBudget(*context, CheckedAdd(csr->ResidentBytes(), CheckedAdd(cursor_bytes, edge_bytes)),
			                           "csr_construction");
		}
		try {
			csr->e.resize(edge_count, 0);
			csr->edge_ids.resize(edge_count, 0);
			csr->write_cursors.reset(new std::atomic<int64_t>[vertex_count]);
		} catch (std::bad_alloc const &) {
			csr->write_failed.store(true);
			throw Exception(ExceptionType::INTERNAL,
			                "Unable to initialize vector of size for csr edge table representation");
		}
		// Keep final offsets immutable; reservations use separate bounded cursors.
		int64_t offset = 0;
		for (size_t src = 0; src < static_cast<size_t>(vertex_count); ++src) {
			const auto degree = csr->v[src + 2].load();
			csr->v[src] = offset;
			csr->write_cursors[src] = offset;
			offset += degree;
		}
		csr->v[vertex_count] = expected;
		csr->v[vertex_count + 1] = expected;
		csr->expected_writes = static_cast<size_t>(expected);
		csr->initialized_e = true;
		csr->phase.store(CSR::Phase::Edges, std::memory_order_release);
		return csr;
	} catch (...) {
		ScheduleDeleteIfOwned(id, csr);
		throw;
	}
}

shared_ptr<CSR> DuckPGQState::InitializeWeights(int32_t id, int64_t edge_count, PhysicalType weight_type,
                                                ClientContext *context) {
	auto csr = GetCSR(id);
	try {
		std::lock_guard<std::mutex> guard(csr->construction_lock);
		if (csr->phase.load() != CSR::Phase::Edges || csr->write_failed.load()) {
			throw ConstraintException("CSR weight construction is not writable");
		}
		if (edge_count < 0 || static_cast<uint64_t>(edge_count) != csr->e.size()) {
			throw InvalidInputException("Invalid CSR weight count");
		}
		if (weight_type != PhysicalType::INT64 && weight_type != PhysicalType::DOUBLE) {
			throw NotImplementedException("Unrecognized weight type detected.");
		}
		const auto mode = weight_type == PhysicalType::INT64 ? CSRWeightMode::Int64 : CSRWeightMode::Double;
		if (csr->mode_set.load(std::memory_order_acquire)) {
			if (csr->weight_mode != mode) {
				throw ConstraintException("Mixed CSR edge weight modes");
			}
			return csr;
		}
		if (context) {
			CheckAlgorithmMemoryBudget(
			    *context,
			    CheckedAdd(csr->ResidentBytes(), CheckedBytes(static_cast<size_t>(edge_count), sizeof(uint64_t))),
			    "csr_weights");
		}
		try {
			if (weight_type == PhysicalType::INT64) {
				csr->w.resize(edge_count, 0);
			} else if (weight_type == PhysicalType::DOUBLE) {
				csr->w_double.resize(edge_count, 0);
			}
		} catch (std::bad_alloc const &) {
			csr->write_failed.store(true);
			throw Exception(ExceptionType::INTERNAL,
			                "Unable to initialize vector of size for csr weight table representation");
		}
		csr->weight_mode = mode;
		csr->initialized_w = true;
		csr->mode_set.store(true, std::memory_order_release);
		return csr;
	} catch (...) {
		ScheduleDeleteIfOwned(id, csr);
		throw;
	}
}

} // namespace duckdb
