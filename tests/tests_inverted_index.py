from __future__ import annotations

from src.engine.inverted_index import PostingList


def test_posting_list_starts_empty() -> None:
    pl = PostingList()
    assert len(pl) == 0
    assert pl.to_list() == []
    assert pl.capacity() == 4


def test_posting_list_append_and_iterate() -> None:
    pl = PostingList()
    pl.append("doc1", 3)
    pl.append("doc2", 1)
    pl.append("doc5", 7)
    assert len(pl) == 3
    assert list(pl) == [("doc1", 3), ("doc2", 1), ("doc5", 7)]


def test_posting_list_grows_with_doubling() -> None:
    pl = PostingList()
    assert pl.capacity() == 4
    for i in range(4):
        pl.append(f"doc{i}", 1)
    assert pl.capacity() == 4   # llena pero no creció todavía
    pl.append("doc4", 1)
    assert pl.capacity() == 8   # doubling
    for i in range(5, 8):
        pl.append(f"doc{i}", 1)
    assert pl.capacity() == 8
    pl.append("doc8", 1)
    assert pl.capacity() == 16


def test_posting_list_preserves_insertion_order() -> None:
    # Si los docs se procesan en orden creciente, la posting list queda ordenada
    # por doc_id sin esfuerzo extra (invariante crítico para multi-way merge).
    pl = PostingList()
    for i in range(20):
        pl.append(f"doc_{i:03d}", i + 1)
    ids = [p[0] for p in pl]
    assert ids == sorted(ids)


def test_posting_list_getitem() -> None:
    pl = PostingList()
    pl.append("a", 1)
    pl.append("b", 2)
    assert pl[0] == ("a", 1)
    assert pl[1] == ("b", 2)
    try:
        _ = pl[2]
    except IndexError:
        pass
    else:
        raise AssertionError("se esperaba IndexError")


def test_posting_list_sort_by_doc_id() -> None:
    pl = PostingList()
    for doc_id, tf in [("doc_c", 1), ("doc_a", 2), ("doc_b", 3)]:
        pl.append(doc_id, tf)
    pl.sort_by_doc_id()
    assert pl.to_list() == [("doc_a", 2), ("doc_b", 3), ("doc_c", 1)]


def test_posting_list_doubling_doesnt_lose_data() -> None:
    # Stress: 1000 appends -> contenido íntegro y orden preservado.
    pl = PostingList()
    expected = [(f"d{i:04d}", i) for i in range(1000)]
    for doc_id, tf in expected:
        pl.append(doc_id, tf)
    assert len(pl) == 1000
    assert pl.to_list() == expected
    # Capacity es potencia de 2 >= 1000
    assert pl.capacity() >= 1000
    assert (pl.capacity() & (pl.capacity() - 1)) == 0


def _run_all() -> None:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} tests pasaron.")


if __name__ == "__main__":
    _run_all()