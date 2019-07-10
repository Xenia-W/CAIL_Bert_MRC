from preprocessing.data_processor import read_squad_data

if __name__ == "__main__":
    read_squad_data("data/all_train_data_augmented.json", "data/")
