def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(tokens), step):
        chunk = tokens[i : i + chunk_size]
        if len(chunk) == 0:
            break
        chunks.append(chunk)

        if i + chunk_size >= len(tokens):
            break

    return chunks
