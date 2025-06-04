from cs336_basics.pretokenization import (
    pretokenize,
) 

def test_run_pretokenize(tmp_path):
    corpus = "Hello world!<|endoftext|>This is a test. test test<|endoftext|>"
    file_path = tmp_path / "sample.txt"
    file_path.write_text(corpus, encoding="utf-8")

    output = pretokenize(file_path, 4)

    print(output)

