from Io.data_loader import create_batch_iter
from preprocessing.data_processor import read_squad_data, convert_examples_to_features, read_qa_examples
from pytorch_pretrained_bert.tokenization import BertTokenizer
from predict.predict import main
from util.plot_util import loss_acc_epoch_plot,loss_acc_f1_step_plot

if __name__ == "__main__":
    # history = {
    #     'train_loss': range(100),
    #     'eval_loss': range(100),
    #     'train_acc': range(100),
    #     'eval_acc': range(100)
    # }
    # loss_acc_epoch_plot(history,"hello.png")
    # loss_acc_f1_step_plot(range(10000),range(10000),range(1000),"omg.png")

    read_squad_data("data/big_train_data.json", "data/")
    # examples = read_qa_examples("data/", "train")
    # print(len(examples))
    # features = convert_examples_to_features(examples,
    #                              tokenizer=BertTokenizer("pretrained_model/vocab.txt"),
    #                              max_seq_length=512,
    #                              doc_stride=500,
    #                              max_query_length=32,
    #                              is_training=True)
    # print(len(features))
    # main()
