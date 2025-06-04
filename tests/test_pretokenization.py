from multiprocessing import Manager
from collections import Counter
from cs336_basics.pretokenization import (
    pretokenize,
    find_chunk_boundaries,
    process_chunk,
)


def _serial_pretokenize(file_path: str, num_processes: int):
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    chunk_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    manager = Manager()
    queue = manager.Queue()
    for chunk_range in chunk_ranges:
        process_chunk(chunk_range, file_path, queue)

    results = []
    while not queue.empty():
        results.append(queue.get())
    return results


def test_pretokenize_matches_serial(tmp_path):
    corpus = "Hello world!<|endoftext|>This is a test.<|endoftext|>"
    file_path = tmp_path / "sample.txt"
    file_path.write_text(corpus, encoding="utf-8")

    parallel_counts = pretokenize(str(file_path), num_processes=2)
    serial_counts = _serial_pretokenize(str(file_path), num_processes=2)

    def _aggregate(counts_list):
        counter = Counter()
        for d in counts_list:
            counter.update(d)
        return counter

    assert _aggregate(parallel_counts) == _aggregate(serial_counts)


def test_find_chunk_boundaries_respects_special_tokens(tmp_path):
    corpus = "Hello<|endoftext|>World<|endoftext|>Bye"
    file_path = tmp_path / "corpus.txt"
    file_bytes = corpus.encode("utf-8")
    file_path.write_bytes(file_bytes)

    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_num_chunks=3, split_special_token=b"<|endoftext|>"
        )

    assert boundaries[0] == 0
    assert boundaries[-1] == len(file_bytes)
    for boundary in boundaries[1:-1]:
        assert file_bytes.startswith(b"<|endoftext|>", boundary)
