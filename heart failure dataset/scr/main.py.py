import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(data):
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    numeric_features = [col for col in data.columns if col not in categorical_features + ['HeartDisease']]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    X = data.drop('HeartDisease', axis=1)
    y = data['HeartDisease']
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor


def train_model(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def plot_confusion_matrix(y_test, y_pred, results_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Disease', 'Disease'],
                yticklabels=['Non-Disease', 'Disease'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()


def plot_classification_report(y_test, y_pred, results_dir):
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)
    labels = ['Non-Disease', 'Disease']

    x = range(len(labels))
    plt.figure(figsize=(8, 6))
    plt.bar(x, precision, width=0.2, label='Precision', align='center')
    plt.bar(x, recall, width=0.2, label='Recall', align='edge')
    plt.bar(x, f1, width=0.2, label='F1-Score', align='edge')
    plt.xticks(x, labels)
    plt.ylim([0, 1])
    plt.title('Classification Metrics')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'classification_report.png'))
    plt.close()


def evaluate_model(model, X_test, y_test, results_dir):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    # Save report to file
    with open(os.path.join(results_dir, 'accuracy_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    # Plot and save evaluation visuals
    plot_confusion_matrix(y_test, y_pred, results_dir)
    plot_classification_report(y_test, y_pred, results_dir)


def save_model(model, model_path):
    joblib.dump(model, model_path)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'heart.csv')
    results_dir = os.path.join(base_dir, 'results')
    models_dir = os.path.join(base_dir, 'models')

    for directory in [results_dir, models_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    data = load_data(data_path)
    X, y, preprocessor = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, results_dir)
    save_model(model, os.path.join(models_dir, 'decision_tree_model.joblib'))


if __name__ == '__main__':
    main()
