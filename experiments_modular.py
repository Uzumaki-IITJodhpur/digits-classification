from utils import DigitClassification

digit_classification = DigitClassification()
digit_classification.load_dataset()
x, y = digit_classification.load_digit_data()
digit_classification.plot_data(x, y, execution="training data")
digit_classification.initiate_svm(gamma=0.001)
x = digit_classification.pre_process(x)
X_train, X_test, y_train, y_test  = digit_classification.split_data(x, y, test_split_ratio=0.3, random_state=42)
digit_classification.train_model(X_train, y_train)
y_test_predicted = digit_classification.predict(X_test)
digit_classification.print_classification_report(y_test_predicted, y_test, "test-set")
digit_classification.display_confusion_matrix(y_test_predicted, y_test, "test-set")

X_train, X_dev, X_test, y_train, y_val, y_test  = digit_classification.split_data_train_dev_test(x, y, dev_size=0.15, test_size=0.15, random_state=42)
y_dev_predicted = digit_classification.predict(X_dev)
digit_classification.print_classification_report(y_dev_predicted, y_test, "dev-set")
digit_classification.display_confusion_matrix(y_dev_predicted, y_test, "dev-set")
print("Complete")