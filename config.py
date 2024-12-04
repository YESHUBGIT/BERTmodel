import transformers

DEVICE = "cpu"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "C:/Users/yeszr/Downloads/BERT/BERTmodel/model.bin"
TRAINING_FILE = "C:/Users/yeszr/Downloads/BERT/BERTmodel/traindata/train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
