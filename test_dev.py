from Io.data_loader import create_batch_iter
from preprocessing.data_processor import read_squad_data, convert_examples_to_features, read_qa_examples
from pytorch_pretrained_bert.tokenization import BertTokenizer
from predict.predict import main
from util.plot_util import loss_acc_epoch_plot,loss_acc_f1_step_plot

if __name__ == "__main__":
    test_data_path = "../data/data.json"
    data_dir = "../result/"

    main(test_data_path, data_dir)
    result = []
    with open("../result/nbest_predictions.json", "r", encoding="utf-8") as fr:
        data = json.load(fr)
    for key, value in data.items():
        res = {"answer": value[0]["text"], "id": key}
        result.append(res)
    with open("../result/result.json", 'w') as fr:
        json.dump(result, fr)
