from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union, TypedDict, Literal
# for models
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from pydantic import BaseModel
from enum import Enum
import datetime as dt

app = FastAPI()


# class WineModelException(Exception):
#     """
#     Base class exception for my ML models server
#     """
#     pass
#
# class WrongModelName(WineModelException):
#     """
#     Raised when trying to use unavailable ML model
#     """
#     def __init__(self, input_model):


class RandomForestParams(BaseModel):
    bootstrap: Union[int, None]
    ccp_alpha: Union[float, None]
    criterion: Union[Literal["squared_error", "absolute_error", "poisson"], None]
    max_depth: Union[int, None]
    min_samples_split: Union[int, float, None]
    min_samples_leaf: Union[int, float, None]
    min_weight_fraction_leaf: Union[float, None]
    max_features: Union[Literal["sqrt", "log2"], int, float, None]
    max_leaf_nodes: Union[int, None]
    min_impurity_decrease: Union[float, None]
    bootstrap: Union[bool, None]
    oob_score: Union[bool, None]
    n_jobs: Union[int, None]
    random_state: Union[int, None]
    verbose: Union[int, None]
    warm_start: Union[bool, None]
    ccp_alpha: Union[float, None]
    max_samples: Union[int, float, None]


class LassoParams(BaseModel):
    alpha: Union[float, None]
    copy_X: Union[int, None]
    fit_intercept: Union[int, None]
    max_iter: Union[int, None]
    positive: Union[int, None]
    precompute: Union[int, None]
    random_state: Union[int, None]
    tol: Union[float, None]
    warm_start: Union[int, None]


class LearningParams(BaseModel):
    model_name: Literal['Lasso', 'RandomForest']
    params: Union[LassoParams, RandomForestParams]
    fit_name: Union[str, None]


df = pd.read_csv('winequality-red.csv')

X = df.drop(['quality'], axis=1)
y = df.quality

ml_dict = {
    'Lasso': {'class': Lasso,
              'models': {}},
    'RandomForest': {'class': RandomForestRegressor,
                     'models': {}}
}


@app.get("/")
def say_hello():
    return "Hello, man! It's server for Artem's ml-ops hw"


@app.get("/available_models")
def available_models():
    """
    :return: list of available classes of models
    """
    return list(ml_dict.keys())


@app.post("/fit_model")
def fit_model(learning_params: LearningParams) -> dict:
    """
    :param learning_params: name of ml model class, hparams and name/id for this fit
    :return: {
             'fit_name': str, # name for using fit via other methods
             'already_fitted': int, # flg if this specification has already been fitted
             }
    """
    model_name = learning_params.model_name
    model_params = learning_params.params.dict(exclude_none=True)
    fit_name = learning_params.fit_name
    exist_flg = 0
    # if model with these hparams has already been fitted
    if len(ml_dict[model_name]['models']) > 0:
        for ex_fit_name in ml_dict[model_name]['models'].keys():
            if model_params == ml_dict[model_name]['models'][ex_fit_name]['params']:
                exist_flg = 1
                exist_name = ex_fit_name
                break
    if exist_flg:
        return {'fit_name': exist_name, 'already_fitted': 1}
    # if not - let's fit one
    if not fit_name:
        fit_name = f"{model_name}-{dt.datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    model = ml_dict[model_name]['class'](**model_params)
    model.fit(X, y)

    ml_dict[model_name]['models'][fit_name] = {'fit': model, 'params': model_params}
    return {'fit_name': fit_name, 'already_fitted': 0}
