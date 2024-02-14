import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, file_path, target):
        self.file_path = file_path
        self.target = target
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path, sep=';')
        self.data["age"] = self.data["age"] / 365.25  # Convert days to years
        self.y = self.data[self.target]
        self.X = self.data.drop(['id', self.target], axis=1)
        return self.X, self.y

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def split_and_scale(self, X, y, test_size=0.2, val_size=0.25):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=0)  # 0.25 x 0.8 = 0.2

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        X_val = self.scaler.transform(X_val)
        return X_train, X_val, X_test, y_train, y_val, y_test

class CardiovascularModel:
    def __init__(self, input_dim):
        self.model = self.build_model(input_dim)

    def build_model(self, input_dim):
        model = Sequential([
            Dense(11, input_shape=(input_dim,), activation='relu'),
            Dense(11, activation='relu'),
            Dense(11, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=1, batch_size=2):
        self.classifier = KerasClassifier(model=self.model, epochs=epochs, batch_size=batch_size, verbose=1)
        self.history = self.classifier.fit(X_train, y_train, validation_data=(X_val, y_val))
        return self.history

    def evaluate(self, X_test, y_test):
        y_pred = self.classifier.predict(X_test)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        return y_pred

class ResultsPlotter:
    @staticmethod
    def plot_results(history_):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history_['accuracy'], color='orange', marker='o', label='Training accuracy')
        plt.plot(history_.history_['val_accuracy'], color='blue', marker='o', label='Validation accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history_.history_['loss'], color='orange', marker='o', label='Training loss')
        plt.plot(history_.history_['val_loss'], color='blue', marker='o', label='Validation loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Main execution
file_path = "dataset/cardio_train.csv"
target = 'cardio'
loader = DataLoader(file_path, target)
X, y = loader.load_data()

preprocessor = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_and_scale(X, y)

model = CardiovascularModel(X_train.shape[1])
history = model.train(X_train, y_train, X_val, y_val)

y_pred = model.evaluate(X_test, y_test)

ResultsPlotter.plot_results(history)
