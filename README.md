# Iris Classification with Kubeflow Pipelines

This project demonstrates a simple, end-to-end machine learning workflow for classifying the Iris dataset, orchestrated using Kubeflow Pipelines (KFP). The pipeline covers data loading, preprocessing, model training, and evaluation.

## Pipeline Overview

The pipeline consists of four sequential components:

1.  **`load_data`**: Loads the classic Iris dataset from `scikit-learn`, converts it into a Pandas DataFrame, and saves it as a CSV artifact.
2.  **`preprocess_data`**: Takes the raw data CSV, splits it into training (80%) and testing (20%) sets, and outputs them as separate CSV artifacts.
3.  **`train_model`**: Trains a `RandomForestClassifier` on the training data and saves the trained model as a `joblib` artifact.
4.  **`evaluate_model`**: Loads the trained model and the test data, evaluates the model's accuracy, and returns the accuracy score as the final output of the pipeline.

## Prerequisites

*   Python 3.10+
*   `pip` for package management
*   Access to a Kubeflow deployment to run the pipeline.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd kfp
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Compile the Pipeline

To use the pipeline, you first need to compile the Python script into a Kubeflow-compatible YAML format.

```bash
python iris_kfp_pipeline.py
```

This command will generate an `iris_kfp_pipeline.yaml` file in your project directory.

### Run the Pipeline

1.  Navigate to your Kubeflow Pipelines dashboard in your web browser.
2.  Create a new pipeline and upload the `iris_kfp_pipeline.yaml` file.
3.  Create a run from the uploaded pipeline to execute the workflow.
4.  Once the run is complete, you can view the logs for each step and see the final accuracy output from the `evaluate-model` component.