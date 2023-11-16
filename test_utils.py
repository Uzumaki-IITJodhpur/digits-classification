from utils import DigitClassification
from itertools import product
import numpy as np

def test_hparam_combination_count():
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    list_of_all_param_combination = [(a, b) for a, b in product(gamma_ranges, C_ranges)]
    assert len(list_of_all_param_combination) == len(gamma_ranges) * len(C_ranges)

def test_hparam_combination_values():
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1]
    list_of_all_param_combination = [(a, b) for a, b in product(gamma_ranges, C_ranges)]

    expected_param_combo_1 = (0.001, 1)
    expected_param_combo_2 = (100, 0.1)

    assert ( expected_param_combo_1 in list_of_all_param_combination ) \
        and ( expected_param_combo_2 in list_of_all_param_combination )

def test_data_split():
    digit_classification = DigitClassification()
    digit_classification.load_dataset()
    x, y = digit_classification.load_digit_data()
    test_split, dev_split = 0.1, 0.2
    train_split = 1 - test_split - dev_split
    X_train, X_dev, X_test, y_train, y_dev, y_test  = digit_classification.split_data_train_dev_test(x, y, dev_size=dev_split, test_size=test_split, random_state=42)
    
    assert (len(X_train) == int(train_split*len(x))) \
        and (len(X_dev) == int(np.ceil(dev_split*len(x)))) \
        and (len(X_test) == int(np.ceil(test_split*len(x))))