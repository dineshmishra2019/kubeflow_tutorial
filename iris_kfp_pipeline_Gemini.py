import kfp
import kfp.dsl as dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model


@dsl.component(
    base_image="python:3.10",
    packages_to_install=['pandas', 'scikit-learn'],
)
def load_data(
    iris_dataset: Output[Dataset]
):
    """Load the Iris dataset and save it to a KFP-managed artifact."""
    import pandas as pd
    from sklearn.datasets import load_iris

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df.to_csv(iris_dataset.path, index=False)


@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'scikit-learn']
)
def preprocess_data(
    input_data: Input[Dataset],
    train_data: Output[Dataset],
    test_data: Output[Dataset]
):
    """Preprocess the data by splitting it into training and testing sets."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_data.path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)


@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'scikit-learn']
)
def train_model(
    train_data: Input[Dataset],
    model_artifact: Output[Model]
):
    """Train a simple model on the training data."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    train_df = pd.read_csv(train_data.path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_artifact.path)


@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def evaluate_model(
    model_artifact: Input[Model],
    test_data: Input[Dataset]
) -> float:
    """Evaluate the model on the test data and return accuracy."""
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import joblib

    test_df = pd.read_csv(test_data.path)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    model = joblib.load(model_artifact.path)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")
    return accuracy


@dsl.pipeline(
    name='iris-kfp-pipeline',
    description='A pipeline that trains and evaluates a model on the Iris dataset.'
)
def iris_kfp_pipeline():
    """A pipeline that orchestrates loading, preprocessing, training, and evaluating."""

    load_task = load_data()
    load_task.set_caching_options(enable_caching=False)  # Disable caching for this step
    # Alternatively, you can disable caching for the entire pipeline run by setting
    # disable_caching=True in the client.create_run_from_pipeline_func() method below.
    # This is useful during development to ensure all steps run fresh.


    # To disable caching for a single step, you can use .set_caching_options(False)
    # For example, to always force data loading:
    # load_task.set_caching_options(enable_caching=False)

    preprocess_task = preprocess_data(input_data=load_task.outputs['iris_dataset'])
    preprocess_task.set_caching_options(enable_caching=False)  # Disable caching for this step

    train_task = train_model(train_data=preprocess_task.outputs['train_data'])
    train_task.set_caching_options(enable_caching=False)  # Disable caching for this step

    eval_task = evaluate_model(model_artifact=train_task.outputs['model_artifact'], test_data=preprocess_task.outputs['test_data'])


    # You can disable caching on any task, including the last one.
    eval_task.set_caching_options(enable_caching=False)


if __name__ == '__main__':
    # It's a good practice to compile the pipeline to a YAML file
    # This file can be used for versioning and for uploading to the KFP UI directly.
    compiler.Compiler().compile(
        pipeline_func=iris_kfp_pipeline,
        package_path='iris_kfp_pipeline_compiled.yaml'
    )

    # (Optional) You can also use the client to run the pipeline from your script
    # This requires the KFP SDK to be configured to talk to a KFP host.
    # For example, you might need to port-forward the KFP UI service:
    # `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80`
    # Then the client can connect to http://localhost:8080
    
    client = kfp.Client(host='http://localhost:8080') # Uncomment and configure your KFP host

    client.create_run_from_pipeline_func(
        iris_kfp_pipeline,
        experiment_name='iris_experiment',
        arguments={},
        # To disable caching for the entire run, set disable_caching to True.
        # disable_caching=True,
    )
    print("Pipeline compiled to iris_kfp_pipeline_compiled.yaml")
    print("Pipeline run created with caching disabled.")
