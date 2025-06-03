import os
from typing import BinaryIO
import regex as re
from multiprocessing import Manager, Pool, Queue, TimeoutError

from typing import Tuple

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(chunk_range: Tuple[int, int], file_path: str, queue: Queue):
    start, end = chunk_range
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    count_map = {}
    for match in re.finditer(PAT, chunk):
        token = match.group()
        count_map[token] = count_map.get(token, 0) + 1

    queue.put(count_map)


def pretokenize(file_path: str, num_processes: int):
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    
    chunk_ranges = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    with Manager() as manager:
        queue = manager.Queue()

        # Bundle args for starmap
        args = [(chunk_range, file_path, queue) for chunk_range in chunk_ranges]

        with Pool() as pool:
            pool.starmap(process_chunk, args)

        # Collect results
        all_counts = []
        while not queue.empty():
            all_counts.append(queue.get())

        return all_counts