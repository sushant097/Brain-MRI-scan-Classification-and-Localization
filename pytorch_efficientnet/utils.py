import torch
import matplotlib.pyplot as plt


def save_model(epochs, model, optimizer, criterion, output_path:str='./output_model.pth'):
    """
    Function to save the trained model
    :param epochs:
    :param model:
    :param optimizer:
    :param criterion:
    :param output_path:
    :return:
    """
    torch.save({
        'epoch':epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss': criterion,
    }, f=output_path)


def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    :param train_acc:
    :param valid_acc:
    :param train_loss:
    :param valid_loss:
    :return:
    """
    # accuracy plots
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"../outputs/accuracy.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"../outputs/loss.png")

