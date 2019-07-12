# ---------- Train -------------------
# ensemble paras
index_ensemble_model = 0

train_file = "train_{}.json".format(index_ensemble_model)
dev_file = "dev_{}.json".format(index_ensemble_model)
output_model_file = "pytorch_model_{}.bin".format(index_ensemble_model)
do_ensemble = True
cross_validation_k = 5
nums_ensemble_models = 7
# model super-paras
LSTM_hidden_size = 256
LSTM_dropout = 0.2

log_path = "output/logs"
plot_path = "output/images/"
data_dir = "data/"
output_dir = "output/checkpoint"
VOCAB_FILE = "pretrained_model/vocab.txt"
bert_model = "pretrained_model/pytorch_pretrained_model"
doc_stride = 128
max_query_length = 100
max_seq_length = 512
do_lower_case = True
train_batch_size = 12
eval_batch_size = 16
learning_rate = 2e-5
num_train_epochs = 6
warmup_proportion = 0.1
no_cuda = False
local_rank = -1
seed = 42
gradient_accumulation_steps = 2
fp16 = False
loss_scale = 0.

answer_type = {"YES": 0, "NO": 1, "no-answer": 2, "long-answer": 3}


# ------------ Predict -----------------
predict_batch_size = 16
n_best_size = 1
max_answer_length = 409
verbose_logging = 1
version_2_with_negative = True
null_score_diff_threshold = 0
