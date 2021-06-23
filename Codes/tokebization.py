
import time
import pandas as pd 
import numpy as np
import pickle
import time
import os
import sys
import logging
from transformers import BertTokenizer


def tokenization_all_datasets(path):
    start = time.time()
    #logging.basicConfig(filename='tokenization.log', level=logging.INFO)
    all_files = os.listdir(path)
    for files in all_files:
        df = pd.read_pickle(path + files)
        print(files)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        df["tokenized"] = df["text"].apply(lambda sent:tokenizer.encode(sent, add_special_tokens=True, 
                                                                              max_length=512, truncation=True,
                                                                              padding='max_length', 
                                                                              return_attention_mask=False))
        df.to_pickle(path + files)
        end = time.time()
        elapsed = end-start
        x = files + "is_done in " + str("%.2f" %(elapsed)) + " seconds\n"
        #logging.info(x)

def main(argv):
    tokenization_all_datasets(argv[0])

if __name__=="__main__":

    main(sys.argv[1:])