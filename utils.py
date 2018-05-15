def get_percentile(diffs, percentile):
    print(int(round(percentile * len(diffs))))
    return diffs[int(round(percentile * len(diffs)))]
