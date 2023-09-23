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
image_combinations = [4, 6, 8]
for itr, image_size in enumerate(image_combinations):
    print(f"Combination Iterations: {itr+1}")
    image_size_str = f"{image_size}x{image_size}"
    test_split, dev_split = (0.2, 0.1)
    # import ipdb; ipdb.set_trace();
    x = digit_classification.image_resize(x, (image_size, image_size))
    X_train, X_dev, X_test, y_train, y_dev, y_test  = digit_classification.split_data_train_dev_test(x, y, dev_size=dev_split, test_size=test_split, random_state=42)
    digit_classification.initiate_svm({"gamma":0.001})
    digit_classification.train_model(X_train=X_train, y_train=y_train)
    train_acc = digit_classification.predict_and_eval(X_train, y_train)
    dev_acc = digit_classification.predict_and_eval(X_dev, y_dev)
    test_acc = digit_classification.predict_and_eval(X_test, y_test)
    # print(f"test_size={test_split} dev_size={dev_split} train_size={1-test_split-dev_split} train_acc={train_acc} dev_acc={dev_acc} test_acc={test_acc} ")
    print(f"image size: {image_size_str}, train_size: {1-test_split-dev_split}, dev_size: {test_split}, train_acc: {train_acc}, dev_acc: {dev_acc}, test_acc: {test_acc}")
