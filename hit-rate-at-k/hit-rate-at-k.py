def hit_rate_at_k(recommendations, ground_truth, k):
    hits = []

    for recs, truth in zip(recommendations, ground_truth):
        top_k = recs[:k]

        # Question:
        # Is ANY item in top_k also in truth?

        if any(item in truth for item in top_k):
            hits.append(1)
        else:
            hits.append(0)

    return sum(hits) / len(hits)