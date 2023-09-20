from pathlib import Path

import matplotlib.pyplot as plt
import pandas
import pandas as pd
import plotly.express as xs
import plotly.graph_objs as go

pd.options.plotting.backend = "plotly"

FILE_NAME = 'data.csv'
X_AXIS_COLUMN = 'Country'
X_AXIS_COLUMN_TITLE = 'Country'
Y_AXIS_COLUMN = 'Valuation ($B)'
Y_AXIS_COLUMN_TITLE = 'Average valuation ($B)'
TITLE = 'StartUp average valuation by country'


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


def line(data: pd.DataFrame):
    """
    Визуализация Line с помощью Plotly
    :param data:
    :return:
    """
    fig = xs.line(data, x=X_AXIS_COLUMN, y=Y_AXIS_COLUMN, markers=True)
    fig.update_layout(
        title=TITLE, title_font_size=20, title_x=0.5,
        xaxis_title=X_AXIS_COLUMN_TITLE, xaxis_title_font_size=16, xaxis_tickfont_size=14,
        yaxis_title=Y_AXIS_COLUMN_TITLE, yaxis_title_font_size=16, yaxis_tickfont_size=14,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_traces(
        line_color="crimson",
        marker=dict(size=12, line=dict(width=2, color='black'), color='white')
    )
    fig.update_xaxes(showgrid=True, gridwidth=2, gridcolor='ivory')
    fig.update_yaxes(showgrid=True, gridwidth=2, gridcolor='ivory')
    fig.show()


def bar_mpl(data: pd.DataFrame):
    """
    Визуализация Bae с помощью matplotlib
    :param data:
    :return:
    """
    plt.bar(data[X_AXIS_COLUMN], data[Y_AXIS_COLUMN], color='blue')
    plt.title(TITLE, fontsize=20)
    plt.xticks(rotation=45)
    plt.xlabel(X_AXIS_COLUMN_TITLE)
    plt.xlabel(Y_AXIS_COLUMN_TITLE)
    plt.grid(True)
    plt.rcParams['axes.axisbelow'] = True
    plt.show()


def pie_mpl(data: pd.DataFrame):
    """
    Визуализация Pie с помощью matplotlib
    :param data:
    :return:
    """
    plt.pie(data[Y_AXIS_COLUMN], labels=data[X_AXIS_COLUMN], autopct='%1.1f%%')
    plt.title(Y_AXIS_COLUMN_TITLE, fontsize=20)
    plt.show()


def line_mpl(data: pd.DataFrame):
    """
    Визуализация Line с помощью matplotlib
    :param data:
    :return:
    """
    plt.plot(data[X_AXIS_COLUMN], data[Y_AXIS_COLUMN], color='crimson', marker='o', markersize=12, mfc='white',
             mec='black')
    plt.title('Average valuation ($B)', fontsize=20)
    plt.xlabel(X_AXIS_COLUMN_TITLE)
    plt.xlabel(Y_AXIS_COLUMN_TITLE)
    plt.grid(True)
    plt.show()


def prepare() -> pd.DataFrame:
    """
    Подготовка данных
    Меняется в зависимости от датасета
    :return: padnas.DataFrame - данные
    """
    path: str = Path.cwd().__str__() + '/'
    file: pd.DataFrame = pd.read_csv(path + FILE_NAME, usecols=[X_AXIS_COLUMN, Y_AXIS_COLUMN])
    cleaned_file = file.dropna(axis=0, how='any')
    cleaned_file[Y_AXIS_COLUMN] = cleaned_file[Y_AXIS_COLUMN].apply(lambda x: float(x[1:]))
    cleaned_file[X_AXIS_COLUMN] = cleaned_file[X_AXIS_COLUMN].apply(lambda x: x[:-1] if x.endswith(',') else x)
    cleaned_file = cleaned_file.groupby(X_AXIS_COLUMN).mean().reset_index()
    return cleaned_file


def scenario():
    """
    Код выполнения сценария
    """
    data = prepare()
    info(data)

    bar(data)
    pie(data)
    line(data)

    bar_mpl(data)
    pie_mpl(data)
    line_mpl(data)


if __name__ == '__main__':
    scenario()
