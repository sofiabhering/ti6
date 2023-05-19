import shutil
import os

from utils import pltDataDistribution

PATH_FA = './datasets/face_age/'
PATH_UTK = './datasets/utk/'
PATH_MERGED = './datasets/merged_ds/'


def processDataSets(plot=False):
    """
    Processa os datasets, separando entre as classes <= 18 e > 18 

    Parametros:
    -----------
    plot:
        Opcional, plota gráfico dos datasets
    """
    dir_fa = os.listdir(PATH_FA)
    dir_utk = os.listdir(PATH_UTK)

    _imgs = {
        "0": [],  # under 18
        "1": []
    }

    for age in dir_fa:
        curr_path = os.path.join(PATH_FA, age)
        key = "0" if int(age) <= 18 else "1"
        _imgs[key].extend(age + '_' + name for name in os.listdir(curr_path))

    if plot:
        pltDataDistribution(_imgs)

    return _imgs


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


def preprocess():
    merged_ds = processDataSets()
    if not os.path.exists(PATH_MERGED):
        os.makedirs('./datasets/merged_ds/0/')
        for i in merged_ds['0']:
            age, filename = i.split('_')
            src = os.path.join(PATH_FA+age+'/', filename)
            shutil.copy(src, './datasets/merged_ds/0/')
        os.makedirs('./datasets/merged_ds/1/')
        for i in merged_ds['1']:
            age, filename = i.split('_')
            src = os.path.join(PATH_FA+age+'/', filename)
            shutil.copy(src, './datasets/merged_ds/1/')


def main():
    preprocess()


if __name__ == '__main__':
    main()
