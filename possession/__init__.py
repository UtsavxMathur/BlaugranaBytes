"""Team Blaugrana Bytes: who has the ball, frame by frame.

    tracking.py     Tracking: one array format for every data source, plus an on-disk cache
    epts.py         reader for FIFA/EPTS tracking (the competition ATD, Metrica Sample Game 3)
    baseline.py     the organisers' baseline on arrays: our known floor and a check of the reader
    features.py     Tracking -> one row of numbers per frame
    trees.py        gradient-boosted trees: export from scikit-learn, exact inference in numpy
    decode.py       frame probabilities -> whole possessions (Viterbi)
    model.py        features + trees + decode = labels
    submission.py   write, validate and fingerprint the CSV

Nothing in this package reads labels of the competition match. Training code lives in training/.
"""
