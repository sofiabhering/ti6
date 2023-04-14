import os
from re import X
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

PATH_FA = './datasets/face_age/'
PATH_UTK = './datasets/utk/'
PATH_UTK_CP = './datasets/utk_cropped/'


def initImgStructure(dir):
    imgs_dic = {}
    images = {}
    for age in dir:
        curr_path = os.path.join(PATH_FA, age)
        num_imgs = len(os.listdir(curr_path))
        images[int(age)] = num_imgs

    pltDataDistribution(imgs_dic.keys(), imgs_dic.values())


def pltDataDistribution(x, y, set_labels=True, save=False, name='Ages x Number of Images'):
    """
    Plota a distribuição de dados de um determinado dataset

    Parametros:
    -----------
    x:
        Dados do eixo X

    y:
        Dados do eixo y

    set_labels:
        Opcional, mostrar labels para cada value do dict

    save:
        Opcional, salva grafico em _graficos/

    name:
        Opcional, nome do grafico

    """
    fig, ax = plt.subplots(figsize=(12, 8))
    _bar = ax.bar(x, y, color='g')

    ax.set(xlabel="Ages", ylabel="Number of Images",
           title=name)

    if set_labels:
        ax.bar_label(_bar, rotation=90, fontsize=8, padding=3, color="#202020")

    plt.savefig(f'./_graficos/barchart_{name}') if save else plt.show()


def combineAgeRanges(plot=False):
    """
    Combina as imagens do PATH_FA em ranges de idade

    Parametros:
    -----------
    dir:
        lista contendo subpastas do PATH_FA
    """
    dir = os.listdir(PATH_FA)
    _dic = {
        "kid": [],
        "teen": [],
        "yadult": [],
        "adult": [],
    }

    for age in dir:
        curr_path = os.path.join(PATH_FA, age)
        i = int(age)
        if i <= 12:
            _dic["kid"].extend(os.listdir(curr_path))
        if i > 12 and i <= 18:
            _dic["teen"].extend(os.listdir(curr_path))
        if i > 18 and i <= 35:
            _dic["yadult"].extend(os.listdir(curr_path))
        if i > 35 and i <= 65:
            _dic["adult"].extend(os.listdir(curr_path))

    if plot:
        _values = []
        for k, v in _dic.items():
            # print(f"{k}: tam = {len(v)}, imagens = {v}")
            _values.append(len(v))

        pltDataDistribution(_dic.keys(), _values, save=True, name="FA_teen20")

    return _dic


def buildUtkDataset(plot=False):
    utk = os.listdir(PATH_UTK)
    # utk_cropped = os.listdir(PATH_UTK_CP)
    _dic = {
        "kid": [],
        "teen": [],
        "yadult": [],
        "adult": [],
    }

    age = utk[0].split('_')[0]
    for file in utk:
        age = file.split('_')[0]
        i = int(age)
        if i <= 12:
            _dic["kid"].append(file)
        if i > 12 and i <= 18:
            _dic["teen"].append(file)
        if i > 18 and i <= 35:
            _dic["yadult"].append(file)
        if i > 35 and i <= 65:
            _dic["adult"].append(file)

    if plot:
        _values = []
        for k, v in _dic.items():
            # print(f"{k}: tam = {len(v)}, imagens = {v}")
            _values.append(len(v))

        pltDataDistribution(_dic.keys(), _values, save=True, name="UTK_teen20")

    return _dic


def mergeDataSets(fa, utk):

    merged_ds = {
        "kid":  utk["kid"],
        "teen": fa["teen"] + utk["teen"],
        "yadult": utk["yadult"],
        "adult": fa["adult"],
    }

    _values = []
    for k, v in merged_ds.items():
        # print(f"{k}: tam = {len(v)}, imagens = {v}")
        _values.append(len(v))

    pltDataDistribution(merged_ds.keys(), _values, save=True, name="merged_ds_teen")

    return merged_ds


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

    for k,v in m_ds.items():
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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    
    print(X_train.shape)
    print(X_train.head())

    print(X_test.shape)
    print(X_test.head())

def main():
    # initImgStructure(dir)
    fa = combineAgeRanges()
    utk = buildUtkDataset()

    m_ds = mergeDataSets(fa, utk)
    createDataFrame(m_ds)




if __name__ == "__main__":
    main()
