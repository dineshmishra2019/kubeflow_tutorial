import kfp
import kfp.dsl as dsl
from kfp.components import create_component_from_func
from kfp import compiler
from typing import NamedTuple, List
import kfp.components as comp


def load_data() -> NamedTuple("Outputs", [("features", List[List[float]]), ("labels", List[int])]):
    from sklearn.datasets import load_iris
    iris = load_iris()
    return (iris.data.tolist(), iris.target.tolist())

load_data_op = comp.create_component_from_func(
    load_data,
    base_image="python:3.10",
    packages_to_install=["pandas", "scikit-learn"]
)

def prepare_data(
    features: List[List[float]],
    labels: List[int]
) -> NamedTuple("Outputs", [("train_features", List[List[float]]), ("test_features", List[List[float]]), ("train_labels", List[int]), ("test_labels", List[int])]):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    return (X_train, X_test, y_train, y_test)
prepare_data_op = comp.create_component_from_func(
    prepare_data,
    base_image="python:3.10",
    packages_to_install=["scikit-learn"]
)
def train_model(
    train_features: List[List[float]],
    train_labels: List[int],
    test_features: List[List[float]],
    test_labels: List[int]
) -> NamedTuple("Output", [("accuracy", float)]):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    clf = RandomForestClassifier()
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)
    acc = accuracy_score(test_labels, y_pred)

    print(f"Model accuracy: {acc}")
    return (acc,)
train_model_op = comp.create_component_from_func(
    train_model,
    base_image="python:3.10",
    packages_to_install=["scikit-learn"]
)

@dsl.pipeline(
    name="iris-no-artifacts-pipeline",
    description="ML pipeline without file artifacts, returns accuracy."
)
def iris_pipeline():
    data = load_data_op()
    prepared_data = prepare_data_op(
        features=data.outputs["features"],
        labels=data.outputs["labels"]
    )
    train_model_op(
        train_features=prepared_data.outputs["train_features"],
        train_labels=prepared_data.outputs["train_labels"],
        test_features=prepared_data.outputs["test_features"],
        test_labels=prepared_data.outputs["test_labels"]
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path="iris_pipeline.yaml"
    )
