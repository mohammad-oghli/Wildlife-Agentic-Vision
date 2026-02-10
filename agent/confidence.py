def update_confidence(previous, deviation_score, persistence):
    increment = deviation_score * min(persistence / 5, 1.0)
    return min(previous + increment, 1.0)
