from utils import DigitClassification
from itertools import product

digit_classification = DigitClassification()
digit_classification.load_dataset()
x, y = digit_classification.load_digit_data()
digit_classification.plot_data(x, y, execution="training data")
x = digit_classification.pre_process(x)
test_splits = [0.1, 0.2, 0.3]
dev_splits = [0.1, 0.2, 0.3]
test_combinations = [(a, b) for a, b in product(test_splits, dev_splits)]
for itr, test_combination in enumerate(test_combinations):
    print(f"Combination Iterations: {itr+1}")
    test_split, dev_split = test_combination
    X_train, X_dev, X_test, y_train, y_dev, y_test  = digit_classification.split_data_train_dev_test(x, y, dev_size=dev_split, test_size=test_split, random_state=42)
    digit_classification.initiate_svm({"gamma":0.001})
    digit_classification.train_model(X_train=X_train, y_train=y_train)
    train_acc = digit_classification.predict_and_eval(X_train, y_train)
    dev_acc = digit_classification.predict_and_eval(X_dev, y_dev)
    test_acc = digit_classification.predict_and_eval(X_test, y_test)
    print(f"test_size={test_split} dev_size={dev_split} train_size={1-test_split-dev_split} train_acc={train_acc} dev_acc={dev_acc} test_acc={test_acc} ")
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    list_of_all_param_combination = [(a, b) for a, b in product(gamma_ranges, C_ranges)]
    _, optimal_gamma, optimal_C = digit_classification.hyper_param_tuning(X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev, X_test=X_test, y_test=y_test, list_of_all_param_combination=list_of_all_param_combination)
    print(f"Best hyper parameters found for split dev:test~{dev_split}:{test_split} -> gamma: {optimal_gamma}, C: {optimal_C}")
    print("\n")