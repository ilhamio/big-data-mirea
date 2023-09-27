import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas
import pandas as pd
import plotly.express as xs
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn.manifold import TSNE
import seaborn as sns
import umap.umap_ as umap

pd.options.plotting.backend = "plotly"

FILE_NAME = 'data.csv'
MNIST_FILE_NAME = 'mnist.csv'
X_AXIS_COLUMN = 'Location'
X_AXIS_COLUMN_TITLE = 'Country'
Y_AXIS_COLUMN = 'OverAll Score'
COLUMNS = ['Location', 'OverAll Score', 'Teaching Score', 'Research Score']
FLOAT_COLUMNS = ['OverAll Score', 'Teaching Score', 'Research Score']
Y_AXIS_COLUMN_TITLE = 'Average score'
TITLE = 'Average overall university score by country'


def info(data: pandas.DataFrame):
    print("Метод info")
    print("==========")
    data.info()
    print("\nМетод head")
    print("==========")
    print(data.head())


def bar(data: pandas.DataFrame):
    """
    Визуализация Bar с помощью Plotly
    :param data:
    :return:
    """
    x = data[X_AXIS_COLUMN]
    y = data[Y_AXIS_COLUMN]
    fig = go.Figure(xs.bar(x=x, y=y, color=x))
    fig.update_traces(marker=dict(line=dict(color='black', width=2)))
    fig.update_layout(
        title=TITLE, title_font_size=20, title_x=0.5,
        xaxis_title=X_AXIS_COLUMN_TITLE, xaxis_title_font_size=16, xaxis_tickfont_size=14,
        yaxis_title=Y_AXIS_COLUMN_TITLE, yaxis_title_font_size=16, yaxis_tickfont_size=14,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_xaxes(tickangle=315)
    fig.show()


def pie(data: pd.DataFrame):
    """
    Визуализация Pie с помощью Plotly
    :param data:
    :return:
    """
    trace = go.Pie(
        labels=data[X_AXIS_COLUMN],
        values=data[Y_AXIS_COLUMN]
    )
    fig = go.Figure(data=trace)
    fig.update_layout(
        title=TITLE, title_font_size=20, title_x=0.5,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_traces(marker=dict(line=dict(color='black', width=2)))
    fig.show()


def line_mpl(data: pd.DataFrame):
    """
    Визуализация Line с помощью matplotlib
    :param data:
    :return:
    """
    plt.figure(figsize=(12, 10))
    plt.plot(data[X_AXIS_COLUMN], data[Y_AXIS_COLUMN], color='crimson', marker='o', markersize=12, mfc='white',
             mec='black')
    plt.plot(data[X_AXIS_COLUMN], data['Teaching Score'], color='crimson', marker='o', markersize=12, mfc='white',
             mec='black')
    plt.title(TITLE, fontsize=20)
    plt.xlabel(X_AXIS_COLUMN_TITLE)
    plt.xticks(rotation=90)
    plt.xlabel(Y_AXIS_COLUMN_TITLE)
    plt.grid(True)
    plt.show()


def clear_float(x: str | float):
    if isinstance(x, str) and '–' in x:
        d = x.split('–')
        return (float(d[0]) + float(d[1])) / 2
    return float(x)


def prepare() -> pd.DataFrame:
    """
    Подготовка данных
    Меняется в зависимости от дтасета
    :return: padnas.DataFrame - данные
    """
    path: str = Path.cwd().__str__() + '/'
    file: pd.DataFrame = pd.read_csv(path + FILE_NAME, usecols=COLUMNS)
    cleaned_file = file.dropna(axis=0, how='any')
    for i in FLOAT_COLUMNS:
        cleaned_file[i] = cleaned_file[i].apply(lambda x: clear_float(x))
    cleaned_file: pd.DataFrame = cleaned_file.groupby(X_AXIS_COLUMN).mean().reset_index()

    return cleaned_file.sort_values(by=[Y_AXIS_COLUMN], ascending=True)[60:]


def prepare_mnist() -> pd.DataFrame:
    path: str = Path.cwd().__str__() + '/'
    data = pd.read_csv(path + MNIST_FILE_NAME)
    scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data))


def mnist(data: pd.DataFrame):
    path: str = Path.cwd().__str__() + '/'
    raw_data = pd.read_csv(path + MNIST_FILE_NAME)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    t = TSNE(n_components=2, perplexity=60, random_state=123)
    tsne_feature = t.fit_transform(data)

    d = data.copy()
    d['x'] = tsne_feature[:, 0]
    d['y'] = tsne_feature[:, 1]

    plt.figure()
    sns.scatterplot(x='x', y='y', hue=raw_data['label'], data=d, palette='bright')
    plt.show()


def umapF(data: pd.DataFrame):
    path: str = Path.cwd().__str__() + '/'
    raw_data = pd.read_csv(path + MNIST_FILE_NAME)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    um = (umap.UMAP(n_neighbors=30, min_dist=0.5, random_state=123).fit_transform(data))

    d = data.copy()
    d['x'] = um[:, 0]
    d['y'] = um[:, 1]

    plt.figure()
    sns.scatterplot(x='x', y='y', hue=raw_data['label'], data=d, palette='bright')
    plt.show()


def scenario():
    """
    Код выполнения сценария
    """
    data = prepare()
    info(data)

    bar(data)
    pie(data)

    line_mpl(data)
    data = prepare_mnist()
    mnist(data)
    umapF(data)


if __name__ == '__main__':
    scenario()
