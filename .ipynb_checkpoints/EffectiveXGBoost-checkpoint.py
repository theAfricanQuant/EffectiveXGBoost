#
#
# (c) Ricky Macharm, MScFE
# https://SisengAI.com
#
#

import os
import pandas as pd
import numpy as np
import urllib.request
import zipfile

from IPython.display import Image, display

from feature_engine import encoding, imputation
from sklearn import base, pipeline, model_selection

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Any, Dict, Union, Sequence

import plotly.graph_objects as go

import xgboost as xgb

import subprocess

def extract_dataset(src_path: str, dst_path: str, dataset: str) -> pd.DataFrame:
    """Extracts a chosen dataset from a zip file and read it into a pandas
    DataFrame.
    
    Parameters:
    ------------
    src_path: str
        URL or path of the zip file to be downloaded.
    dst_path: str
        Local file path where the zip file will be written.
    dataset: str
        Name of the particular file inside the zip file to be 
        read into a DataFrame.
    
    Returns:
    ---------
    pandas.core.frame.DataFrame: 
        DataFrame containing the contents of the selected dataset.
        
    """
    
    # using `urllib.request` module to open the URL specified in `src_path`.    
    open_path = urllib.request.urlopen(src_path)

    # reads the contents of the file object `open_path`
    data = open_path.read()

    # using context management to write the file in the destination path or folder
    with open(dst_path, mode='wb') as f:
        f.write(data)

    # using context management to extract the saved zip file.
    with zipfile.ZipFile(dst_path) as z:
        all_df = pd.read_csv(z.open(dataset))
        df_ = all_df.iloc[0]
        return all_df.iloc[1:]
    
    

def topx(pser: pd.Series, x: int = 5, default='other') -> pd.Series:
    """Replace all values in a Pandas Series that are not among
    the top `x` most frequent values with a default value.

    Parameters
    ----------
    pser : pd.Series
        The input pandas Series.
    x : int, optional
        The number of most frequent values to keep. The
        default value is 5.
    default : str, optional
        The default value to use for values that are not among
        the top `x` most frequent values. The default value is
        'other'.

    Returns:
    ---------
    pandas.Series: 
        Series containing the results.
    """

    val_count = pser.value_counts()
    return (pser.where(pser
                      .isin(val_count.index[:x]),
                      default)
           )


def prep_data(pdf: pd.DataFrame) -> pd.DataFrame:
    """ This function prepares the rurvey data and returns a new DataFrame.
    The prepatations include extracting and transforming certain
    columns, renaming some, and selecting a subset of columns.
    
    Parameters
    ----------
    pdf : pd.DataFrame
        The input DataFrame containing the survey data.
    
    Returns
    -------
    pd.DataFrame:
        The new DataFrame with the modified and selected columns.
    """
    
    return (pdf
            .assign(age=(pdf['Q2']
                         .str.slice(0,2)
                         .astype(int)
                        ),
                   education=(pdf
                             ['Q4']
                             .replace({'Doctoral degree' : 20, 'Bachelor’s degree' : 16, 
                                       'Master’s degree' : 18,
                                   'Professional degree' : 19,
                                   'Some college/university study without earning a bachelor’s degree': 13,
                                   'I prefer not to answer' : None, 
                                    'No formal education past high school': 12})
                            ),
                    major=(pdf['Q5']
                           .pipe(topx, x=3)
                           .replace({'Engineering (non-computer focused)': 'eng',
                                       'Computer science (software engineering, etc.)': 'cs',
                                       'Mathematics or statistics' : 'stat',})
                          ),
                    years_exp=(pdf
                             ['Q8']
                             .str.replace('+', '', regex=False)
                             .str.split('-', expand=True)
                             .iloc[:,0]
                             .astype(float)
                            ),
                    compensation=(pdf
                             ['Q9']
                             .str.replace('+', '', regex=False)
                             .str.replace(',','')
                             .replace({'500000':'500',
                                      'I do not wish to disclose my approximate yearly compensation':'0'})
                             .str.split('-', expand=True)
                             .iloc[:,0]
                             .fillna(0)
                             .astype(int)
                             .mul(1_000)
                            ),
                    python=(pdf
                             ['Q16_Part_1']
                             .fillna(0)
                             .replace({'Python': 1})
                            ),
                    r=(pdf
                         ['Q16_Part_2']
                         .fillna(0)
                         .replace({'R': 1})
                        ),
                    sql=(pdf
                         ['Q16_Part_3']
                         .fillna(0)
                         .replace({'SQL': 1})
                        )
                    )
            .rename(columns=lambda col:col.replace(' ', '_'))
            .loc[:, 'Q1,Q3,age,education,major,years_exp,compensation,python,r,sql'.split(',')]
    )


class PrepDataTransformer(base.BaseEstimator,
    base.TransformerMixin):
    """
    This transformer takes a Pandas DataFrame containing our survey 
    data as input and returns a new version of the DataFrame. 
    
    ----------
    ycol : str, optional
        The name of the column to be used as the target variable.
        If not specified, the target variable will not be set.
    Attributes
    ----------
    ycol : str
        The name of the column to be used as the target variable.
    """
    def __init__(self, ycol=None):
        self.ycol = ycol
    
    def transform(self, X):
        return prep_data(X)

    def fit(self, X, y=None):
        return self

def prepX_y(df, y_col):
    raw = (df
    .query('Q3.isin(["United States of America", "China", "India"]) '
    'and Q6.isin(["Data Scientist", "Software Engineer"])')
    )
    return raw.drop(columns=[y_col]), raw[y_col]


def my_dot_export(model, n_trees, filename, title='', direction='TB'):
    """Exports a specified number of trees from an XGBoost model as a graph
    visualization in dot and png formats.
    Parameters:
    -----------
    model: 
        An XGBoost model.
    n_trees: 
        The number of tree to export.
    filename: 
        The name of the file to save the exported visualization.
    title: 
        The title to display on the graph visualization (optional).
    direction: 
        The direction to lay out the graph, either 'TB' (top to
        bottom) or 'LR' (left to right) (optional).
    """
    res = xgb.to_graphviz(model, num_trees=n_trees)
    content = f''' node [fontname = "Roboto Condensed"];
    edge [fontname = "Roboto Thin"];
    label = "{title}"
    fontname = "Roboto Condensed"
    '''
    out = res.source.replace('graph [ rankdir=TB ]',
                             f'graph [ rankdir={direction} ];\n {content}')
    # dot -Gdpi=300 -Tpng -ocourseflow.png courseflow.dot
    dot_filename = filename
    with open(dot_filename, 'w') as f:
        f.write(out)
    png_filename = dot_filename.replace('.dot', '.png')
    subprocess.run(f'dot -Gdpi=300 -Tpng -o{png_filename} {dot_filename}'.split())
    
    
def inv_logit(p: float) -> float:
    """
    Compute the inverse logit function of a given value.
    The inverse logit function is defined as:
    f(p) = exp(p) / (1 + exp(p))
    Parameters
    ----------
    p : float
        The input value to the inverse logit function.
    
    Returns
    -------
    float
        The output of the inverse logit function.
    """
    return np.exp(p) / (1 + np.exp(p))

def my_image_export(model, n_trees, filename, title='', direction='TB'):
    """
    Export a specified number of trees from an XGBoost model as a graph
    visualization in dot and png formats.

    Parameters:
    -----------
    model : xgboost.core.Booster
        The XGBoost model to visualize.
    n_trees : int
        The number of trees to export.
    filename : str
        The name of the file to save the exported visualization.
    title : str, optional
        The title to display on the graph visualization.
    direction : str, optional
        The direction to lay out the graph. Valid values are 'TB' (top to bottom)
        and 'LR' (left to right).

    Returns:
    --------
    None

    Notes:
    ------
    This function generates a dot file containing the graph visualization of the
    specified number of trees from the model. It then modifies the dot file to add
    a title and set the direction of the graph layout. The modified dot file is saved
    to disk as a dot file and a png file. The png file has the same name as the dot file
    but with the '.png' extension.

    Example:
    --------
    >>> import xgboost as xgb
    >>> model = xgb.train(params, dtrain)
    >>> my_dot_export(model, n_trees=2, filename='mytree', title='My Tree Visualization', direction='LR')

    This example exports the first two trees from the specified XGBoost model as a
    graph visualization with the title 'My Tree Visualization' and a left-to-right
    layout. The visualization is saved to disk as 'mytree.dot' and 'mytree.png'.
    """
    res = xgb.to_graphviz(model, num_trees=n_trees)
    content = f''' node [fontname = "Roboto Condensed"];
    edge [fontname = "Roboto Thin"];
    label = "{title}"
    fontname = "Roboto Condensed"
    '''
    out = res.source.replace('graph [ rankdir=TB ]',
                             f'graph [ rankdir={direction} ];\n {content}')
    # dot -Gdpi=300 -Tpng -ocourseflow.png courseflow.dot
    dot_filename = filename
    with open(dot_filename, 'w') as f:
        f.write(out)
    png_filename = dot_filename.replace('.dot', '.png')
    subprocess.run(f'dot -Gdpi=300 -Tpng -o{png_filename} {dot_filename}'.split())
    display(Image(filename=png_filename))

    
def hyperparameter_tuning(space: Dict[str, Union[float, int]],
X_train: pd.DataFrame, y_train: pd.Series,
X_test: pd.DataFrame, y_test: pd.Series,
early_stopping_rounds: int=50,
metric:callable=accuracy_score) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning for an XGBoost classifier.
    This function takes a dictionary of hyperparameters, training
    and test data, and an optional value for early stopping rounds,
    and returns a dictionary with the loss and model resulting from
    the tuning process. The model is trained using the training
    data and evaluated on the test data. The loss is computed as
    the negative of the accuracy score.
    Parameters
    ----------
    space : Dict[str, Union[float, int]]
    A dictionary of hyperparameters for the XGBoost classifier.
    X_train : pd.DataFrame
    The training data.
    y_train : pd.Series
    The training target.
    X_test : pd.DataFrame
    The test data.
    y_test : pd.Series
    The test target.
    early_stopping_rounds : int, optional
    The number of early stopping rounds to use. The default value
    is 50.
    metric : callable
    Metric to maximize. Default is accuracy
    Returns
    -------
    Dict[str, Any]
    A dictionary with the loss and model resulting from the
    tuning process. The loss is a float, and the model is an
    XGBoost classifier.
    """
    int_vals = ['max_depth', 'reg_alpha']
    space = {k: (int(val) if k in int_vals else val)
    for k,val in space.items()}
    space['early_stopping_rounds'] = early_stopping_rounds
    model = xgb.XGBClassifier(**space)
    evaluation = [(X_train, y_train),
    (X_test, y_test)]
    model.fit(X_train, y_train,
    eval_set=evaluation,
    verbose=False)
    pred = model.predict(X_test)
    score = metric(y_test, pred)
    return {'loss': -score, 'status': STATUS_OK, 'model': model}


def plot_3d_mesh(df: pd.DataFrame, x_col: str, y_col: str,
    z_col: str) -> go.Figure:
    """
    Create a 3D mesh plot using Plotly.
    This function creates a 3D mesh plot using Plotly, with
    the `x_col`, `y_col`, and `z_col` columns of the `df`
    DataFrame as the x, y, and z values, respectively. The
    plot has a title and axis labels that match the column
    names, and the intensity of the mesh is proportional
    to the values in the `z_col` column. The function returns
    a Plotly Figure object that can be displayed or saved as
    desired.
    Parameters
    ----------
    df : pd.DataFrame
    The DataFrame containing the data to plot.
    x_col : str
    The name of the column to use as the x values.
    y_col : str
    The name of the column to use as the y values.
    z_col : str
    The name of the column to use as the z values.
    Returns
    -------
    go.Figure
    A Plotly Figure object with the 3D mesh plot.
    """
    fig = go.Figure(data=[go.Mesh3d(x=df[x_col], y=df[y_col], z=df[z_col],
                                    intensity=df[z_col]/ df[z_col].min(),
                                    hovertemplate=f"{z_col}: %{{z}}<br>{x_col}: %{{x}}<br>{y_col}: "
                                    "%{{y}}<extra></extra>")],
    )
    fig.update_layout(
        title=dict(text=f'{y_col} vs {x_col}'),
        scene = dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col),
        width=700,
        margin=dict(r=20, b=10, l=10, t=50)
            )
    return fig


def jitter(df: pd.DataFrame, col: str, amount: float=1) -> pd.Series:
    """
    Add random noise to the values in a Pandas DataFrame column.
    This function adds random noise to the values in a specified
    column of a Pandas DataFrame. The noise is uniform random
    noise with a range of `amount` centered around zero. The
    function returns a Pandas Series with the jittered values.
    Parameters
    ----------
    df : pd.DataFrame
    The input DataFrame.
    col : str
    The name of the column to jitter.
    amount : float, optional
    The range of the noise to add. The default value is 1.
    Returns
    -------
    pd.Series
    A Pandas Series with the jittered values.
    """

    vals = np.random.uniform(low=-amount/2, high=amount/2,
    size=df.shape[0])
    return df[col] + vals


def trial2df(trial: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a Trial object (sequence of trial dictionaries)
    to a Pandas DataFrame.
    Parameters
    ----------
    trial : List[Dict[str, Any]]
    A list of trial dictionaries.
    Returns
    -------
    pd.DataFrame
    A DataFrame with columns for the loss, trial id, and
    values from each trial dictionary.
    """
    vals = []
    for t in trial:
        result = t['result']
        misc = t['misc']
        val = {k:(v[0] if isinstance(v, list) else v)
            for k,v in misc['vals'].items()
            }
        val['loss'] = result['loss']
        val['tid'] = t['tid']
        vals.append(val)
    return pd.DataFrame(vals)