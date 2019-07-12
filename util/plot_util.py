import matplotlib.pyplot as plt
import config.args as config
import os

# 无图形界面需要加，否则plt报错
plt.switch_backend('agg')


def loss_acc_epoch_plot(history, filename = "loss_acc_epoch.png"):
    train_loss = history['train_loss']
    eval_loss = history['eval_loss']
    train_accuracy = history['train_acc']
    eval_accuracy = history['eval_acc']

    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(2, 1, 1)
    plt.title('loss during train')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss)
    plt.plot(epochs, eval_loss)
    plt.legend(['train_loss', 'eval_loss'])

    fig.add_subplot(2, 1, 2)
    plt.title('accuracy during train')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_accuracy)
    plt.plot(epochs, eval_accuracy)
    plt.legend(['train_acc', 'eval_acc'])

    plt.savefig(os.path.join(config.plot_path, filename))


def loss_acc_f1_step_plot(step_loss, step_acc, step_f1, filename="loss_acc_f1_step.png"):
    train_loss = step_loss
    train_accuracy = step_acc
    train_f1 = step_f1

    fig = plt.figure(figsize=(20, 20))
    fig.add_subplot(3, 1, 1)
    plt.title('loss during train')
    plt.xlabel('steps')
    plt.ylabel('loss')
    steps = range(1, len(train_loss) + 1)
    plt.plot(steps, train_loss)
    plt.legend(['train_loss'])

    fig.add_subplot(3, 1, 2)
    plt.title('accuracy during train')
    plt.xlabel('steps')
    plt.ylabel('accuracy')
    steps = range(1, len(train_loss) + 1)
    plt.plot(steps, train_accuracy)
    plt.legend(['train_acc'])

    fig.add_subplot(3, 1, 3)
    plt.title('f1 during train')
    plt.xlabel('steps')
    plt.ylabel('f1')
    steps = range(1, len(train_f1) + 1)
    plt.plot(steps, train_f1)
    plt.legend(['train_f1'])

    plt.savefig(os.path.join(config.plot_path, filename))


if __name__ == '__main__':
    history = {
        'train_loss': range(100),
        'eval_loss': range(100),
        'train_accuracy': range(100),
        'eval_accuracy': range(100)
    }
    loss_acc_epoch_plot(history)
