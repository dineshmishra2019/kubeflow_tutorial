import kfp
from kfp import dsl
from kfp import components
from kfp import compiler
from typing import NamedTuple, List, Any

@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'sklearn']
)
def load_data() -> NamedTuple('Outputs', [('data', str)]):
    """Load the Iris dataset and save it as a CSV file."""
    import pandas as pd
    from sklearn.datasets import load_iris
    import os

    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    output_path = '/tmp/iris.csv'
    df.to_csv(output_path, index=False)
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['data'])
    return output(data=output_path)
@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'sklearn']
)
def preprocess_data(data: str) -> NamedTuple('Outputs', [('train_data', str), ('test_data', str)]):
    """Preprocess the data by splitting it into training and testing sets."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os

    df = pd.read_csv(data)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_path = '/tmp/train.csv'
    test_path = '/tmp/test.csv'
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['train_data', 'test_data'])
    return output(train_data=train_path, test_data=test_path)

@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'sklearn']
)
def train_model(train_data: str) -> NamedTuple('Outputs', [('model', str)]):
    """Train a simple model on the training data."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    import os

    train_df = pd.read_csv(train_data)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    model_path = '/tmp/model.joblib'
    joblib.dump(model, model_path)
    
    from collections import namedtuple
    output = namedtuple('Outputs', ['model'])
    return output(model=model_path)

@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'sklearn']
)
def evaluate_model(model: str, test_data: str) -> float:
    """Evaluate the model on the test data and return accuracy."""
    import pandas as pd
    from sklearn.metrics import accuracy_score
    import joblib

    test_df = pd.read_csv(test_data)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    model = joblib.load(model)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")
    return accuracy

@dsl.pipeline(
    name='iris-kfp-pipeline',
    description='A pipeline that trains and evaluates a model on the Iris dataset.'
)
def iris_kfp_pipeline() -> float:
    """A pipeline that orchestrates loading, preprocessing, training, and evaluating."""
    load_task = load_data()
    preprocess_task = preprocess_data(data=load_task.outputs['data'])
    train_task = train_model(train_data=preprocess_task.outputs['train_data'])
    eval_task = evaluate_model(model=train_task.outputs['model'], test_data=preprocess_task.outputs['test_data'])
    
    return eval_task.output 

if __name__ == '__main__':
    compiler.Compiler().compile(iris_kfp_pipeline, 'iris_kfp_pipeline.yaml')

