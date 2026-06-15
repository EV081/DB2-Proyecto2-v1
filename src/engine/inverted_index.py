from __future__ import annotations

from typing import Iterator

Posting = tuple[str, int] #(doc_id, tf)


class PostingList:
    __slots__ = ("_data", "_size")

    _INITIAL_CAPACITY = 4

    def __init__(self) -> None:
        self._data: list[Posting | None] = [None] * self._INITIAL_CAPACITY
        self._size = 0

    def append(self, doc_id: str, tf: int) -> None:
        if self._size == len(self._data):
            self._grow()
        self._data[self._size] = (doc_id, tf)
        self._size += 1

    def _grow(self) -> None:
        new_cap = len(self._data) * 2
        new_data: list[Posting | None] = [None] * new_cap
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Posting]:
        for i in range(self._size):
            yield self._data[i]

    def __getitem__(self, i: int) -> Posting:
        if i < 0 or i >= self._size:
            raise IndexError(f"posting index {i} out of range [0, {self._size})")
        return self._data[i]

    def capacity(self) -> int:
        return len(self._data)

    def to_list(self) -> list[Posting]:
        return [self._data[i] for i in range(self._size)] 

    def sort_by_doc_id(self) -> None:
        sorted_view = sorted(
            (self._data[i] for i in range(self._size)),
            key=lambda p: p[0],
        )
        for i, p in enumerate(sorted_view):
            self._data[i] = p


__all__ = ["Posting", "PostingList"]
