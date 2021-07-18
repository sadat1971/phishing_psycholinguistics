import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import tqdm
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup
import random
import torch.nn.functional as F
import sys
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, classes, dropout=0.25):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, classes)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

def create_dataloader(train_text, train_labels, batch_size=16):
    # Create the DataLoader for our training set
    '''
    This function will create a dataloader for our training set. The dataloader will help to feed the randomly 
    sampled data on each batch. The batch size is selected to be 16
    '''
    train_data = TensorDataset(torch.tensor(train_text), torch.tensor(train_labels))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

def initialize_model(model, epochs, train_dataloader,
                     learning_rate=1e-5,):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return optimizer, scheduler

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, epochs, device, optimizer, scheduler):
    """Train the BertClassifier model.
    """
    loss_fn = nn.CrossEntropyLoss()
    # Start training loop
    print("Start training...\n")
    
    for epoch_i in range(epochs):

      #For better visulization
      print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
      print("-"*70)

      # start timer
      t0_epoch, t0_batch = time.time(), time.time()

      #reset the loss
      total_loss, batch_loss, batch_counts = 0, 0, 0

      #set the training mode
      model.train()

      #start training in batches
      for batch_no, batch in enumerate(train_dataloader):
        batch_counts += 1

        train_X, train_y = tuple(t.to(device) for t in batch)

        #zero out the gradient
        model.zero_grad()

        #forward pass
        output = model(train_X.float())


        #compute the loss
        loss = loss_fn(output, train_y)
        batch_loss += loss.item()
        total_loss += loss.item()

        #performing backward pass
        loss.backward()

        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and the learning rate
        optimizer.step()
        scheduler.step()

          # Print the loss values and time elapsed for every 20 batches
        if (batch_no % 100 == 0 and batch_no != 0) or (batch_no == len(train_dataloader) - 1):
            # Calculate time elapsed for 20 batches
            time_elapsed = time.time() - t0_batch

            # Print training results
            print(f"{epoch_i + 1:^7} | {batch_no:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

            # Reset batch tracking variables
            batch_loss, batch_counts = 0, 0
            t0_batch = time.time()


        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        #print("-"*10)
    return model

def predict(model, test_X, test_y, batch_size, device, labels_there=True):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    test_dataset = TensorDataset(torch.tensor(test_X))
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    all_logits = []

    # For each batch in our test set...
    for batch in (test_dataloader):
        # Load batch to GPU
        b_input_ids= batch[0].to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids.float())
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    df = pd.DataFrame()
    if labels_there==True:
        df["GT"] = test_y
    df["probs_1"] = probs[:,1]
    df["prediction"] = df["probs_1"].apply(lambda x:1 if x>=0.5 else 0)
    return df

def main(argv):

    
    hidden_size = int(argv[0])
    dropout = float(argv[1])
    batch_size = int(argv[2])
    epochs = int(argv[3])
    hyp_search = (argv[4])
    labelname = argv[5]


        #first, we'll see if we have CUDA available
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    set_seed(42)

    saved_path = "/disk2/sadat/PhishingResearch/processed_work/psych_trait_experiments/Files/"
    full_train_set_X = np.load("/disk2/sadat/PhishingResearch/processed_work/psych_trait_experiments/Files/SBERT_Soft_labels_combined.npy")
    full_train_set = pd.read_pickle("/disk2/sadat/PhishingResearch/processed_work/psych_trait_experiments/Files/Soft_labels_combined.pkl")
    full_train_set["Urgency"] = full_train_set["Urgency"].apply(lambda x:1 if x>=1 else 0)
    full_train_set_y = full_train_set["Urgency"].values
    if hyp_search=="yes":
        
        for RS in [42, 43, 44]:
            train_set_X, test_X, train_set_y, test_y = train_test_split(full_train_set_X, full_train_set_y, test_size=0.20, random_state=RS)
            train_dataloader = create_dataloader(train_set_X, train_set_y, batch_size=batch_size)
            model = NeuralNet(input_size=768, hidden_size=hidden_size, classes=2, dropout=dropout).to(device)
            optimizer, scheduler = initialize_model(model=model, epochs=epochs, train_dataloader=train_dataloader)
            model = train(model, train_dataloader, epochs=epochs, device=device, optimizer=optimizer, scheduler=scheduler)
            df = predict(model=model, test_X=test_X, test_y=test_y, batch_size=batch_size, device=device)
            accuracy = accuracy_score(df["GT"], df["prediction"])
            f1 = f1_score(df["GT"], df["prediction"])
            with open(saved_path+"results.txt", 'a') as r:
                res = "testSet: " + str(RS)+  "   hidden:" + argv[0] + "  drop:" + argv[1]+ "  batch_size:" +  argv[2] + "   epchs:" +  argv[3] + "  **F1 Score:" + str(f1) + "  **Accuracy" + str(accuracy)
                r.write(res)
                r.write("\n\n")
        
    else:
        train_dataloader = create_dataloader(full_train_set_X, full_train_set_y, batch_size=batch_size)
        model = NeuralNet(input_size=768, hidden_size=hidden_size, classes=2, dropout=dropout).to(device)
        optimizer, scheduler = initialize_model(model=model, epochs=epochs, train_dataloader=train_dataloader)
        model = train(model, train_dataloader, epochs=epochs, device=device, optimizer=optimizer, scheduler=scheduler)
        for dataset in ["SBERT_combined_iwspa_ap_train_set.npy", "SBERT_iwspa_ap_test.npy", "SBERT_spamSMS.npy"]:
            test_npy = np.load(saved_path + dataset)
            df = predict(model=model, test_X=test_npy, test_y=None, batch_size=batch_size, device=device, labels_there=False)
            pkl_file = dataset[6:-4] + ".pkl"
            orig_dataset = pd.read_pickle(saved_path+pkl_file)
            orig_dataset[labelname+"_SBERT"] = df["probs_1"]
            orig_dataset.to_pickle(saved_path + pkl_file)

if __name__=="__main__":
    main(sys.argv[1:])