import kfp
from kfp import dsl
from kfp import components
from kfp import compiler
# import kfp.dsl as dsl
# from kfp.components import create_component_from_function
# from kfp.v2 import compiler
from typing import NamedTuple, List, Any
import json
import os

@dsl.component
def hello_kfp(name: str) -> str:
    """A simple component that greets the user."""
    greeting = f"Hello, {name}!"
    print(greeting)
    return greeting

@dsl.pipeline(
    name='hello-kfp-pipeline',
    description='A simple pipeline that greets the user.'
)

def hello_kfp_pipeline(name: str = 'KFP User') -> str:
    """A simple pipeline that uses the hello_kfp component."""
    hello_task = hello_kfp(name=name)
    return hello_task.output

if __name__ == '__main__':
    compiler.Compiler().compile(hello_kfp_pipeline, 'hello_kfp_pipeline.yaml')


