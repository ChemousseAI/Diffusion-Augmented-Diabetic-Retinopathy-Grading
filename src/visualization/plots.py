
import matplotlib.pyplot as plt

def plot_history(train_loss, train_acc):
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Loss')
    ax.plot(train_acc, label='Accuracy')
    ax.legend()
    return fig
