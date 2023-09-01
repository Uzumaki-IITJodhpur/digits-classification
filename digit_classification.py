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
    
    def initiate_svm(self, gamma):
        self.clf = svm.SVC(gamma=gamma)

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
        x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_split_ratio, shuffle=False,random_state=random_state
        )

        return x_train, x_test, y_train, y_test 

    def train_model(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x_test=None):
        return self.clf.predict(x_test)
    
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
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Calculate the remaining size for the development set
        remaining_size = 1.0 - test_size
        dev_relative_size = dev_size / remaining_size

        # Split the train data into train and development subsets
        x_train, x_dev, y_train, y_dev = train_test_split(
            x_train, y_train, test_size=dev_relative_size, random_state=random_state
        )

        return x_train, x_dev, x_test, y_train, y_dev, y_test
