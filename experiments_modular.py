from digit_classification import DigitClassification

digit_classification = DigitClassification()
digit_classification.load_dataset()
digit_classification.load_digit_data()
digit_classification.plot_data()
digit_classification.initiate_svm(gamma=0.001)
digit_classification.pre_process()
x_train, x_test, y_train, y_test  = digit_classification.split_data()
digit_classification.train_model()
y_predicted = digit_classification.predict()
digit_classification.print_classification_report(y_predicted, y_test)
digit_classification.display_confusion_matrix(y_predicted, y_test)
print("Complete")