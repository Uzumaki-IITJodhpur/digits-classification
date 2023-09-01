import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

class DigitClassification:
    def __init__(self) -> None:
        self.model = None
        self.datasets = None
        self.digits = None
        self.data = None
        self.clf = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        None

    def load_dataset(self):
        self.datasets = datasets

    def load_digit_data(self):
        self.digits = self.datasets.load_digits()
    
    def initiate_svm(self, gamma):
        self.clf = svm.SVC(gamma=gamma)

    def plot_data(self, x, y, execution=""):
        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, label in zip(axes, x, y):
            ax.set_axis_off()
            image = image.reshape(8, 8)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title(f"{execution}: %i" % label)

    def pre_process(self):
        n_samples = len(self.digits.images)
        self.data = self.digits.images.reshape((n_samples, -1))

    def split_data(self, test_split_ratio):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        self.data, self.digits.target, test_size=test_split_ratio, shuffle=False,random_state=random_state
        )

        return self.x_train, self.x_test, self.y_train, self.y_test 

    def train_model(self):
        self.clf.fit(self.x_train, self.y_train)

    def predict(self, x_test=None):
        if type(x_test) != type(None):
            return self.clf.predict(self.x_test)
    
    def print_classification_report(self, y_predicted, y_true):
        print(
            f"Classification report for classifier {self.clf}:\n"
            f"{metrics.classification_report(y_predicted, y_true)}\n"
        )
    
    def display_confusion_matrix(self, y_predicted, y_true):
        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_predicted)
        disp.figure_.suptitle("Confusion Matrix")
        print(f"Confusion matrix:\n{disp.confusion_matrix}")
        plt.show()
        y_true = []
        y_pred = []
        cm = disp.confusion_matrix

        for gt in range(len(cm)):
            for pred in range(len(cm)):
                y_true += [gt] * cm[gt][pred]
                y_pred += [pred] * cm[gt][pred]
        
        self.print_classification_report(y_true, y_pred)

    def split_train_dev_test(X, y, test_size, dev_size, random_state=1):
        # Split data into train and test subsets
        self.X_train, self.X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Calculate the remaining size for the development set
        remaining_size = 1.0 - test_size
        dev_relative_size = dev_size / remaining_size

        # Split the train data into train and development subsets
        X_train_final, X_dev, y_train_final, y_dev = train_test_split(
            X_train, y_train, test_size=dev_relative_size, random_state=random_state
        )

        return X_train_final, X_dev, X_test, y_train_final, y_dev, y_test

    # def predict_and_eval(model, X_test, y_test):
    #     print(
    #     f"Classification report for classifier {model}:\n"
    #     f"{metrics.classification_report(y_test, X_test)}\n"
    #     )
