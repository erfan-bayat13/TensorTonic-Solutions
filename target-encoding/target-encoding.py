def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    # dictionaries for sum and count
    sum_dict = {}
    count_dict = {}

    # accumulate sums and counts
    for i in range(len(categories)):
        cat = categories[i]
        y = targets[i]
        if cat in sum_dict:
            sum_dict[cat] += y
            count_dict[cat] += 1
        else:
            sum_dict[cat] = y
            count_dict[cat] = 1

    # compute mean for each category
    mean_dict = {}
    for cat in sum_dict:
        mean_dict[cat] = sum_dict[cat] / count_dict[cat]

    # replace categories with their mean target
    encoded = [mean_dict[cat] for cat in categories]

    return encoded
