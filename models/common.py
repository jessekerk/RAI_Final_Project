from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data():
    # load dateset

    german_credit = fetch_ucirepo(id=144)

    X = german_credit.data.features.copy()
    y = german_credit.data.targets.iloc[:, 0].copy()

    print("Dataset shape:", X.shape)
    print("\nOriginal target distribution:")
    print(y.value_counts())

    # convert good/bad credit score
    y = y.map({1: 1, 2: 0})

    return X, y, german_credit.variables


def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
