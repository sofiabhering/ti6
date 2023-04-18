import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

PATH_FA = './datasets/face_age/'
PATH_UTK = './datasets/utk/'


def plotNumImgsPerAge():

    def get_age(filename):
        return int(filename.split('_')[0])

    images = {}

    dir_fa = os.listdir(PATH_FA)
    dir_utk = sorted(os.listdir(PATH_UTK), key=get_age)

    for i in range(17):
        age = dir_fa[i]
        curr_path = os.path.join(PATH_FA, age)
        num_imgs = len(os.listdir(curr_path))
        images[int(age)] = num_imgs

    for file in dir_utk:
        age = int(file.split('_')[0])
        if age < 18:
            images[int(age)] += 1

    pltDataDistribution(images)


def pltDataDistribution(dic, set_labels=True, save=False, name='Ages x Number of Images'):
    """
    Plota a distribuição de dados de um determinado dataset

    Parametros:
    -----------
    dic:
        Dict contendo imgs

    set_labels:
        Opcional, mostrar labels para cada value do dict

    save:
        Opcional, salva grafico em _graficos/

    name:
        Opcional, nome do grafico

    """

    _values = []
    for k, v in dic.items():
        # print(f"{k}: tam = {len(v)}, imagens = {v}")
        _values.append(len(v))
        # _values.append(v)
    x = dic.keys()
    y = _values

    fig, ax = plt.subplots(figsize=(12, 8))
    _bar = ax.bar(x, y, color='g')

    ax.set(xlabel="Ages", ylabel="Number of Images",
           title=name)

    if set_labels:
        ax.bar_label(_bar, rotation=90, fontsize=8, padding=3, color="#202020")

    plt.savefig(f'./_graficos/barchart_{name}') if save else plt.show()


def mergeDataSets(plot=False):
    """
    Combina os datasets em um só, até idade 18

    Parametros:
    -----------
    plot:
        Opcional, plota gráfioco dos datasets
    """
    dir_fa = os.listdir(PATH_FA)
    dir_utk = os.listdir(PATH_UTK)

    _imgs = {i: [] for i in range(1, 19)}

    for i in range(18):
        folder = dir_fa[i]
        curr_path = os.path.join(PATH_FA, folder)
        _imgs[int(folder)].extend(os.listdir(curr_path))

    for file in dir_utk:
        age = int(file.split('_')[0])
        if age <= 18:
            _imgs[age].append(file)

    if plot:
        pltDataDistribution(_imgs)

    return _imgs


def createDataFrame(m_ds):

    def _key_correspondente(k):
        if (k == "kid"):
            return 0
        elif k == "teen":
            return 1
        elif k == "yadult":
            return 2
        elif k == "adult":
            return 3

    # inverter rows e values do merged_ds
    m_ds_invertido = {}

    for k, v in m_ds.items():
        for i in v:
            m_ds_invertido[i] = _key_correspondente(k)

    # add coluna de idade
    df = pd.DataFrame.from_dict(m_ds_invertido, orient='index')
    df = pd.DataFrame()
    df["file"] = m_ds_invertido.keys()
    df["class"] = m_ds_invertido.values()
    # print(df)

    x = df["file"]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=42)

    print(X_train.shape)
    print(X_train.head())

    print(X_test.shape)
    print(X_test.head())


def main():
    merged_ds = mergeDataSets(True)
    createDataFrame(merged_ds)


if __name__ == "__main__":
    main()
