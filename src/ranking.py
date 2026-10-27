"""Rank independent parent contexts rather than counting overlapping children twice."""


def best_per_parent(items, score_key):
    best = {}
    for item in items:
        child = item["chunk"]
        key = (child["file_name"], child.get("parent_id", str(child["chunk_id"])))
        if key not in best or item[score_key] > best[key][score_key]:
            best[key] = item
    return sorted(best.values(), key=lambda item: (-item[score_key], str(item["chunk"]["chunk_id"])))
