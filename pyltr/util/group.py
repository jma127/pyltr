"""

Utilities for grouping queries together.

Query ids (qids) can generally be any hashable type (except NoneType), although
int is preferred.

"""


def check_qids(qids):
    """Asserts that query ids are grouped into contiguous blocks.

    Note that query ids do not have to be sorted.

    Parameters
    ----------
    qids : array_like of shape = [n_samples]
        List of query ids.

    Returns
    -------
    int
        Number of unique query ids in `qids`.

    Raises
    ------
    ValueError
        If any two query ids are not in the same contiguous block.
        e.g. ``[1, 1, 3, 3, 2, 2, 3]``

    """
    seen_qids = set()
    prev_qid = None

    for qid in qids:
        assert qid is not None
        if qid != prev_qid:
            if qid in seen_qids:
                raise ValueError('Samples must be grouped by qid.')
            seen_qids.add(qid)
            prev_qid = qid

    return len(seen_qids)


def get_groups(qids):
    """Makes an iterator of query groups on the provided list of query ids.

    Parameters
    ----------
    qids : array_like of shape = [n_samples]
        List of query ids.

    Yields
    ------
    row : (qid, int, int)
        Tuple of query id, from, to.
        ``[i for i, q in enumerate(qids) if q == qid] == range(from, to)``

    """
    prev_qid = None
    prev_limit = 0
    total = 0

    for i, qid in enumerate(qids):
        total += 1
        if qid != prev_qid:
            if i != prev_limit:
                yield (prev_qid, prev_limit, i)
            prev_qid = qid
            prev_limit = i

    if prev_limit != total:
        yield (prev_qid, prev_limit, total)
