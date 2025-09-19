import kfp
from kfp import dsl
from kfp.components import create_component_from_function
from kfp.v2 import compiler
from typing import NamedTuple, List
import json
import os

@dsl.component(
    base_image='python:3.10',
    packages_to_install=['pandas', 'sklearn']

)