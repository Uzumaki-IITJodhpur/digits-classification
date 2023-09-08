import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

class DigitClassification:
    def __init__(self) -> None:
        self.model = None
        self.datasets = None
        self.digits = None
        self.clf = None

    def load_dataset(self):
        self.datasets = datasets

    def load_digit_data(self):
        self.digits = self.datasets.load_digits()
        return self.digits.images, self.digits.target
    
    def initiate_svm(self, model_param):
        if type(model_param) != dict:
            self.clf = svm.SVC(gamma=model_param)
        else:
            self.clf = svm.SVC(**model_param)

    def plot_data(self, x, y, execution=""):
        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, label in zip(axes, x, y):
            ax.set_axis_off()
            image = image.reshape(8, 8)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title(f"{execution}: %i" % label)

    def pre_process(self, data):
        n_samples = len(data)
        return self.digits.images.reshape((n_samples, -1))

    def split_data(self, x, y, test_split_ratio, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_split_ratio, shuffle=False,random_state=random_state
        )

        return X_train, X_test, y_train, y_test 

    def train_model(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test=None):
        return self.clf.predict(X_test)
    
    def predict_and_eval(self, X, y):
        predicted = self.clf.predict(X)    
        return metrics.accuracy_score(y, predicted)

    
    def print_classification_report(self, y_predicted, y_true, data_name=""):
        print(
            f"Classification report for classifier {self.clf}:\n"
            f"on {data_name} " if data_name != "" else "" + "\n"
            f"{metrics.classification_report(y_predicted, y_true)}\n"
        )
    
    def display_confusion_matrix(self, y_predicted, y_true, data_name=""):
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_predicted)
        disp.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        print(f"on {data_name} " if data_name != "" else "" + "\n")
        plt.show()
        y_true = []
        y_pred = []
        cm = disp.confusion_matrix

        for gt in range(len(cm)):
            for pred in range(len(cm)):
                y_true += [gt] * cm[gt][pred]
                y_pred += [pred] * cm[gt][pred]
        
        self.print_classification_report(y_true, y_pred)

    def split_data_train_dev_test(self, X, y, dev_size, test_size, random_state=42):
        # Split data into train and test subsets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Calculate the remaining size for the development set
        remaining_size = 1.0 - test_size
        dev_relative_size = dev_size / remaining_size

        # Split the train data into train and development subsets
        X_train, X_dev, y_train, y_dev = train_test_split(
            X_train, y_train, test_size=dev_relative_size, random_state=random_state
        )

        return X_train, X_dev, X_test, y_train, y_dev, y_test
    
    def hyper_param_tuning(self, X_train, y_train, X_dev, y_dev, X_test, y_test, list_of_all_param_combination):
        """
        ================================
        Recognizing hand-written digits
        ================================

        This example shows how scikit-learn can be used to recognize images of
        hand-written digits, from 0-9.

        """

        # Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
        # License: BSD 3 clause


        # gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
        # C_ranges = [0.1, 1, 2, 5, 10]


        # HYPER PARAMETER TUNING
        # - take all combinations of gamma and C
        best_acc_so_far = -1
        best_model, optimal_gamma, optimal_C = None, None, None
        
        for param_combination in list_of_all_param_combination:
                # print("Running for gamma={} C={}".format(cur_gamma, cur_C))
                # - train model with cur_gamma and cur_C
                # # 5. Model training
                gamma, C = param_combination
                self.initiate_svm({'gamma': gamma, 'C': C})
                self.train_model(X_train, y_train)
                # - get some performance metric on DEV set
                cur_accuracy = self.predict_and_eval(X_dev, y_dev)
                # - select the hparams that yields the best performance on DEV set
                if cur_accuracy > best_acc_so_far:
                    # print("New best accuracy: ", cur_accuracy)
                    best_acc_so_far = cur_accuracy
                    optimal_gamma = gamma
                    optimal_C = C
                    best_model = self.model
        # print("Optimal parameters gamma: ", optimal_gamma, "C: ", optimal_C)


        # 6. Getting model predictions on test set
        # 7. Qualitative sanity check of the predictions
        # 8. Evaluation
        test_acc = self.predict_and_eval(X_test, y_test)
        print("Test accuracy: ", test_acc)
        return best_model, optimal_gamma, optimal_C
