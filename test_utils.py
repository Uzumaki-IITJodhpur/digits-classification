from itertools import product

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