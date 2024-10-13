import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import random
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, classification_report
from time import time


class ClassificationModel:
    """
    A class to train and predict error types from the 3W dataset using a customizable model.
    
    Attributes
    ----------
    dataset_path : str
        Directory where the parquet files are stored.
    model : Any
        Custom model passed to the class.
    scaler : StandardScaler
        Scaler used to normalize the feature data.

    Methods
    -------
    load_and_prepare_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        Loads, cleans, and prepares data from a parquet file.
    split_files(files: List[str], train_size: float = 0.7) -> Tuple[List[str], List[str]]:
        Splits the files into training and testing sets.
    load_data_from_files(files: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        Loads and concatenates data from a list of files.
    train_model(X_train: pd.DataFrame, y_train: pd.Series) -> None:
        Trains the provided model using the training data.
    predict_single_file(file_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        Predicts classes for a single file.
    plot_predictions_with_timestamp(timestamps: pd.Series, X: pd.DataFrame, 
                                    y_true: pd.Series, y_pred: pd.Series, y_axis: str) -> None:
        Plots predictions and actual class values against timestamps.
    complete_analysis(y_axis: str, files: List[str], train_size: float = 0.7) -> None:
        Performs the complete analysis from data loading to prediction and visualization.
    """

    def __init__(self, dataset_path: str, model: Any):
        """
        Initializes the ClassificationModel class with the specified model, and data directory.

        Parameters
        ----------
        model : Any
            Custom model to be used for training and prediction.
        dataset_path : str, optional
            Directory where the parquet files are stored.
        """
        self.dataset_path = dataset_path
        self.model = model
        self.scaler = StandardScaler()

    def load_and_prepare_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load, clean, and prepare data from a parquet file."""
        df = pd.read_parquet(file_path)
        print(f"Loaded {file_path} with {len(df)} rows.")

        df.reset_index(inplace=True)
        df = df.dropna(subset=["timestamp"]).drop_duplicates(subset="timestamp").fillna(0)
        df = df.sort_values(by="timestamp")

        timestamps = df["timestamp"]
        X = df.drop(['class', 'state', 'timestamp'], axis=1)
        y = df['class'].fillna(0).astype(int)

        return X, y, timestamps

    def split_files(self, files: List[str], train_size: float = 0.7) -> Tuple[List[str], List[str]]:
        """Split the files into training and testing sets."""
        random.shuffle(files)
        split_idx = int(len(files) * train_size)
        return files[:split_idx], files[split_idx:]

    def load_data_from_files(self, files: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and concatenate data from a list of files."""
        X_list, y_list = [], []
        for file in files:
            X, y, _ = self.load_and_prepare_data(file)
            X_list.append(X)
            y_list.append(y)

        X_all = pd.concat(X_list, axis=0).reset_index(drop=True)
        y_all = pd.concat(y_list, axis=0).reset_index(drop=True)

        return X_all, y_all

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the provided model using the training data."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict_single_file(self, file_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Predict classes for a single file."""
        X, y, timestamps = self.load_and_prepare_data(file_path)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        return X, y, y_pred, timestamps

    def plot_predictions_with_timestamp(self, timestamps: pd.Series, X: pd.DataFrame, 
                                        y_true: pd.Series, y_pred: pd.Series, y_axis: str) -> None:
        """Plot predictions and actual class values against timestamps."""
        y_true_normalized = self.normalize_classes(y_true)
        y_pred_normalized = self.normalize_classes(y_pred)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=timestamps, y=X[y_axis], mode='lines', name=y_axis, line=dict(color='blue')
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps, y=y_true_normalized, mode='markers', name='Actual Class',
                marker=dict(color='green', size=8, symbol='circle'),
                text=[f"Class: {c}" for c in y_true]
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps, y=y_pred_normalized, mode='markers', name='Predicted Class',
                marker=dict(color='red', size=8, symbol='x'),
                text=[f"Pred: {p}" for p in y_pred]
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=f'{y_axis} with Actual vs Predicted Class (Normalized)',
            xaxis_title='Timestamp',
            yaxis_title=y_axis,
            legend=dict(x=0, y=1.1, orientation='h'),
        )

        # Update secondary y-axis with tick values dynamically based on unique classes
        unique_classes = sorted(y_true.unique())
        tick_vals = self.normalize_classes(pd.Series(unique_classes))
        tick_text = [f'Class {c}' for c in unique_classes]

        fig.update_yaxes(
            title_text="Class Values (Normalized)", secondary_y=True,
            tickvals=tick_vals, ticktext=tick_text
        )

        fig.show()

    def normalize_classes(self, classes: np.ndarray) -> List[float]:
        """
        Normalize class values for clearer visualization by distributing 
        them equally across a range [0, 100].

        Parameters
        ----------
        classes : np.ndarray
            Array of class values to be normalized.

        Returns
        -------
        List[float]
            List of normalized class values equally spaced across [0, 100].
        """
        # Ensure the classes are treated as a Pandas Series
        classes_series = pd.Series(classes)

        # Get the unique class values and sort them
        unique_classes = sorted(classes_series.unique())

        # Generate equally spaced values between 0 and 100
        normalized_values = np.linspace(0, 100, len(unique_classes))

        # Create a mapping from original class to normalized value
        class_mapping = {cls: norm for cls, norm in zip(unique_classes, normalized_values)}

        # Map the original classes to normalized values
        return [class_mapping[c] for c in classes]


    def complete_analysis(self, y_axis: str, files: List[str], train_size: float = 0.7) -> None:
        """
        Perform a complete analysis by training the model, predicting, and visualizing results.

        Parameters
        ----------
        y_axis : str
            The feature to plot on the y-axis.
        files : List[str]
            List of parquet files to analyze.
        train_size : float, optional
            Proportion of data to use for training (default is 0.7).
        """
        train_files, test_files = self.split_files(files, train_size)
        print(f"Training on {len(train_files)} files, Testing on {len(test_files)} files.")

        X_train, y_train = self.load_data_from_files(train_files)
        self.train_model(X_train, y_train)

        for file in test_files:
            X, y_true, y_pred, timestamps = self.predict_single_file(file)
            print(classification_report(y_true, y_pred, zero_division=0))
            self.plot_predictions_with_timestamp(timestamps, X, y_true, y_pred, y_axis)

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Evaluate a model's performance using several metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth (actual class labels).
        y_pred : np.ndarray
            Predicted class labels.

        Returns
        -------
        dict
            Dictionary containing accuracy and F1-score.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }

    def compare_models(self, models: List[Tuple[str, Any]], 
                       train_files: List[str], 
                       test_files: List[str]) -> pd.DataFrame:
        """
        Train and evaluate multiple models to compare their performance.

        Parameters
        ----------
        models : List[Tuple[str, Any]]
            List of tuples with model names and model objects.
        train_files : List[str]
            List of files used for training.
        test_files : List[str]
            List of files used for testing.

        Returns
        -------
        pd.DataFrame
            DataFrame with evaluation metrics for each model.
        """
        results = []

        # Load and prepare the training data
        X_train, y_train = self.load_data_from_files(train_files)

        for model_name, model in models:
            print(f"Training model: {model_name}")
            start_time = time()

            # Train the model
            self.model = model
            self.train_model(X_train, y_train)

            # Collect predictions on the test files
            y_true_all, y_pred_all = [], []
            for file in test_files:
                _, y_true, y_pred, _ = self.predict_single_file(file)
                y_true_all.extend(y_true)
                y_pred_all.extend(y_pred)

            # Evaluate the model's performance
            metrics = self.evaluate_model(np.array(y_true_all), np.array(y_pred_all))
            metrics["model"] = model_name
            metrics["train_time"] = time() - start_time

            results.append(metrics)

        # Create a DataFrame to compare results
        results_df = pd.DataFrame(results)

        # Display the results
        print("\nModel Comparison:")
        print(results_df[["model", "accuracy", "f1_score", "train_time"]])

        return results_df

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight='balanced')

    dataset_path = './dataset/1'

    # Initialize the classification model
    classification_model = ClassificationModel(dataset_path=dataset_path, model=model)

    # Get the list of parquet files
    all_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".parquet")]

    # Shuffle and limit to a subset of files for testing
    all_files = all_files[:10]

    # Run the complete analysis
    classification_model.complete_analysis(y_axis='P-MON-CKP', files=all_files)
