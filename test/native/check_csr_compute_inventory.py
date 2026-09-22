#!/usr/bin/env python3
"""Pin every CSR compute consumer and the two intentional metadata readers."""
import re
import sys
from pathlib import Path

COMPUTE = {
    "adamic_adar": {"adamic_adar"},
    "article_rank": {"article_rank"},
    "betweenness_centrality": {"betweenness_centrality"},
    "cheapest_path_length": {"cheapest_path_length"},
    "closeness_centrality": {"closeness_centrality"},
    "common_neighbors": {"common_neighbors"},
    "cosine_similarity": {"cosine_similarity"},
    "degree_centrality": {"degree_centrality"},
    "eccentricity": {"eccentricity"},
    "eigenvector_centrality": {"eigenvector_centrality"},
    "global_clustering_coefficient": {"global_clustering_coefficient"},
    "harmonic_centrality": {"harmonic_centrality"},
    "hits": {"hits_authority", "hits_hub"},
    "in_degree_centrality": {"in_degree_centrality"},
    "iterativelength": {"iterativelength"},
    "iterativelength2": {"iterativelength2"},
    "iterativelength_bidirectional": {"iterativelengthbidirectional"},
    "jaccard_similarity": {"jaccard_similarity"},
    "k_core_decomposition": {"k_core_decomposition"},
    "katz_centrality": {"katz_centrality"},
    "label_propagation": {"label_propagation"},
    "local_clustering_coefficient": {"local_clustering_coefficient"},
    "louvain": {"louvain"},
    "out_degree_centrality": {"out_degree_centrality"},
    "overlap_similarity": {"overlap_similarity"},
    "pagerank": {"pagerank"},
    "personalized_pagerank": {"personalized_pagerank"},
    "preferential_attachment": {"preferential_attachment"},
    "reachability": {"reachability"},
    "resource_allocation": {"resource_allocation"},
    "shortest_path": {"shortestpath"},
    "single_source_shortest_path": {"single_source_shortest_path"},
    "strongly_connected_component": {"strongly_connected_component"},
    "topological_sort": {"topological_sort"},
    "triangle_count": {"triangle_count"},
    "weakly_connected_component": {"weakly_connected_component"},
}
RAW = {
    "src/core/functions/scalar/csr_get_w_type.cpp",
    "src/core/functions/function_data/cheapest_path_length_function_data.cpp",
}
CONTROL = {
    "csr_creation": {"create_csr_vertex", "create_csr_edge"},
    "csr_deletion": {"delete_csr"},
    "csr_get_w_type": {"csr_get_w_type"},
}


def source(text):
    # Preserve quoted literals while removing comments; neither can fake a call.
    return re.sub(
        r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|/\*.*?\*/|//[^\n]*',
        lambda match: match.group(1) or "",
        text,
        flags=re.S,
    )


def check(root):
    errors, reads, finalizers, owner_finalizers = [], {}, {}, {}
    scalar = root / "src/core/functions/scalar"
    files = {path.stem: path for path in scalar.glob("*.cpp")}
    expected = dict(COMPUTE, **CONTROL)
    if set(files) != set(expected):
        errors.append("scalar source inventory changed: " + str(sorted(set(files) ^ set(expected))))
    for stem, names in expected.items():
        if stem not in files:
            continue
        text = source(files[stem].read_text())
        actual = set(re.findall(r'\bScalarFunction(?:Set(?:\s+\w+)?)?\s*\(\s*"([^"]+)"', text))
        if actual != names:
            errors.append(stem + ": registered API names changed")
    for path in (root / "src").rglob("*.cpp"):
        text = source(path.read_text())
        name = path.relative_to(root).as_posix()
        calls = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', "", text)
        raw_count = len(re.findall(r'(?:->|\.)\s*GetCSR\s*\(', calls))
        finalize_count = len(re.findall(r'(?:->|\.)\s*FinalizeCSRForRead\s*\(', calls))
        owner_finalize_count = len(re.findall(r'(?:->|\.)\s*FinalizeForRead\s*\(', calls))
        if owner_finalize_count:
            owner_finalizers[name] = owner_finalize_count
        if raw_count:
            reads[name] = raw_count
        if finalize_count:
            finalizers[name] = finalize_count
    compute_paths = {"src/core/functions/scalar/" + stem + ".cpp" for stem in COMPUTE}
    if reads != {name: 1 for name in compute_paths | RAW}:
        errors.append("CSR acquisition inventory changed")
    if finalizers != {name: 1 for name in compute_paths}:
        errors.append("each compute site must finalize exactly once; metadata readers must not finalize")
    if owner_finalizers != {"src/duckpgq_state.cpp": 1}:
        errors.append("only the state-bound finalizer may enter the owner read boundary")
    assert len(COMPUTE) == 36 and sum(map(len, COMPUTE.values())) == 37
    return errors


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parents[2]
    errors = check(root)
    for error in errors:
        print("CSR_INVENTORY_FAILED " + error, file=sys.stderr)
    if errors:
        raise SystemExit(1)
    print("CSR_COMPUTE_INVENTORY_ASSERTED compute_sites=36 scalar_apis=37 raw_readers=2")
