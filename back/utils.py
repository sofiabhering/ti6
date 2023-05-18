import matplotlib.pyplot as plt
import os


def plot_results(history, epochs_range):
    '''
    Plota um gráfico com os resultados do modelo
    '''
    print('-------Salvando gráfico-------')
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    os.makedirs('_graficos', exist_ok=True)
    plt.savefig('_graficos/experiment_7030_rgb.jpg')


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
