from __future__ import annotations

from pathlib import Path

from src.data.qrels import load_beir_qrels_tsv


def test_load_beir_qrels_tsv(tmp_path: Path) -> None:
    qrels_path = tmp_path / "test.tsv"
    qrels_path.write_text(
        "query-id\tcorpus-id\tscore\n"
        "q1\td1\t1\n"
        "q1\td2\t0\n"
        "q2\td9\t2\n",
        encoding="utf-8",
    )

    qrels = load_beir_qrels_tsv(qrels_path, min_relevance=1).qrels
    assert qrels == {"q1": {"d1"}, "q2": {"d9"}}

