// Run under AddressSanitizer. The legacy switch changes ONLY setup/deletion
// syntax so this identical acquisition/use-after-removal probe can run against
// the pre-repair API. Old GetCSR returns an unowned pointer and ASan must fail;
// the repaired API retains the same payload until the acquired handle drops.
#include "duckpgq_state.hpp"
using namespace duckdb;
int main() {
    DuckPGQState state;
#ifdef DUCKPGQ_LEGACY_CSR
    auto initial = make_uniq<CSR>();
    initial->vsize = 18;
    initial->v = new atomic<int64_t>[18];
    initial->v[0] = 73;
    initial->initialized_v = true;
    state.csr_list.emplace(0, std::move(initial));
#else
    state.InitializeVertex(0, 16)->v[0] = 73;
#endif
    auto acquired = state.GetCSR(0);
#ifdef DUCKPGQ_LEGACY_CSR
    state.csr_list.erase(0);
#else
    state.DeleteCSR(0);
#endif
    // No other owner survives on the legacy side. This is the actual
    // production GetCSR result, not a model of the ownership implementation.
    return acquired->v[0].load() == 73 ? 0 : 1;
}
