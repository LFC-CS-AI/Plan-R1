"""Validate atlas referential integrity; uses only Python stdlib and no model/data."""
from pathlib import Path
from collections import Counter, deque
import csv
import re

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT / name).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        result = list(reader)
        assert result, f"Empty CSV: {name}"
        assert all(None not in r and None not in r.values() for r in result), name
        return result


def unique(items, key):
    values = [r[key] for r in items]
    assert len(values) == len(set(values)), f"Duplicate {key}"
    return set(values)


def main():
    papers, leads, refs, relations, nodes, methods = [
        read(f"{n}.csv") for n in
        ("papers", "leads", "references", "relations", "nodes", "methods")
    ]
    paper_ids = unique(papers, "id")
    lead_ids = unique(leads, "lead_id")
    node_ids = unique(nodes, "node_id")
    unique(refs, "reference_id")
    unique(relations, "relation_id")
    assert len(papers) == 16 and sum(int(r["quota"]) for r in papers) == 15
    assert {r["id"] for r in papers if r["quota"] == "0"} == {"R00"}
    assert all(r["read_level"] == "L3" for r in papers)
    assert all(r["quota"] == "0" and r["read_level"] != "L3" for r in leads)
    core_status = Counter(r["publication_status"] for r in papers if r["quota"] == "1")
    assert core_status == {"formal": 14, "preprint": 1}, core_status
    assert paper_ids.isdisjoint(lead_ids)
    assert {n["node_id"] for n in nodes if n["node_type"] == "Paper"} == paper_ids | lead_ids
    assert {n["node_type"] for n in nodes} == {
        "Paper", "Problem", "Method", "Assumption", "Limitation", "Benchmark"
    }
    assert unique(methods, "paper_id") == paper_ids
    for m in methods:
        for field in ("problems", "methods", "assumptions", "limitations", "benchmarks"):
            assert set(m[field].split(";")) <= node_ids, (m["paper_id"], field)
    required_fields = [
        "title", "authors", "year", "venue", "publication_url", "fulltext_url",
        "read_version", "research_problem", "prior_limitation", "core_method",
        "key_innovation", "assumptions", "evidence", "baselines",
        "author_limitations", "open_questions", "relation_to_root", "reproduction",
        "checked_sections", "verified_on"
    ]
    for p in papers:
        assert all(p[k].strip() for k in required_fields), p["id"]
        note = (ROOT / "paper_notes" / f"{p['id']}.md").read_text(encoding="utf-8")
        assert len(re.findall(r"^## \d+\.", note, re.M)) == 12, p["id"]
        assert "作者：" in note and "Limitations" in note and "Open Questions" in note
        assert p["fulltext_url"] in note and p["title"] in note
    assert {p.stem for p in (ROOT / "paper_notes").glob("*.md")} == paper_ids
    ref_pairs = [(r["citing_id"], r["cited_id"]) for r in refs]
    assert len(ref_pairs) == len(set(ref_pairs))
    for r in refs:
        assert r["citing_id"] in paper_ids | lead_ids and r["cited_id"] in paper_ids | lead_ids
        assert r["verification"] == "verified_primary_text" and r["locator"]
        assert r["source_url"].startswith("https://") and r["source_version"]
    rel_types = {r["relation"] for r in relations}
    assert {"CITES", "EXTENDS", "ADDRESSES", "USES", "COMPETES_WITH", "LIMITED_BY", "POTENTIALLY_COMPLEMENTS"} <= rel_types
    for r in relations:
        assert r["source_id"] in node_ids and r["target_id"] in node_ids
        assert all(r[k].strip() for k in ("evidence_status", "source_url", "locator", "evidence_or_rationale"))
        if r["relation"] == "EXTENDS":
            assert r["evidence_status"] == "author_explicit"
        if r["relation"] == "POTENTIALLY_COMPLEMENTS":
            assert r["evidence_status"] == "hypothesis"
        if r["relation"] == "COMPETES_WITH":
            assert r["evidence_status"] in {"verified_comparison", "inferred"}
    graph_cites = {(r["source_id"], r["target_id"]) for r in relations if r["relation"] == "CITES"}
    assert graph_cites == set(ref_pairs)
    # Discovery reachability is undirected; CITES direction remains intact in files.
    graph = {p: set() for p in paper_ids | lead_ids}
    for a, b in ref_pairs:
        graph[a].add(b)
        graph[b].add(a)
    distance, todo = {"R00": 0}, deque(["R00"])
    while todo:
        a = todo.popleft()
        for b in graph[a]:
            if b not in distance:
                distance[b] = distance[a] + 1
                todo.append(b)
    assert all(distance[p] <= 2 for p in paper_ids | lead_ids)
    for md in ROOT.rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)", md.read_text(encoding="utf-8")):
            if link.startswith(("http://", "https://", "#")):
                continue
            assert (md.parent / link.split("#")[0]).exists(), (md, link)
    assert not any(p.suffix.lower() in {".pdf", ".ckpt", ".pt", ".db"} for p in ROOT.rglob("*"))
    print(f"PASS: 15 core + Root; {len(leads)} partial lead; {len(refs)} citations; {len(nodes)} nodes; {len(relations)} relations")
    print("Core publication status:", dict(core_status))
    print("Relation types:", dict(Counter(r["relation"] for r in relations)))


if __name__ == "__main__":
    main()
