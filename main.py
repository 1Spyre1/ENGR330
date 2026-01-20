import os
import requests
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import logging
import re
import sklearn

from time import time
from tqdm.auto import tqdm
from torch import nn
from datasets import load_dataset
from datetime import datetime, timezone, timedelta
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class FireDetectionModule():
    def __init__(self, data_path, out_dir, split, batch_size=8, version="v1",device="cpu", gpus="0,1,2,3", epochs=5, lr=0.001, weight_decay=0.001):
        self.dataPath = data_path
        self.outDir = out_dir

        self.split = split

        self.batchSize = batch_size

        self.device = device
        self.gpus = gpus

        self.epochs = epochs
        self.lr = lr
        self.wd = weight_decay

        self.bestLoss = 100
        self.bestAcc = 0
        if os.path.exists(f"models/{version}"):
            for file in os.listdir(f"models/{version}"):
                try:
                    loss = float(file.split("_")[-3])
                    acc = float(file[:-4].split("_")[-1])
                    if loss <= self.bestLoss and acc >= self.bestAcc:
                        self.bestLoss = loss
                        self.bestAcc = acc

                except:
                    pass
                    
        gmt3 = timezone(timedelta(hours=3))
        date = datetime.now(gmt3)
        date = date.strftime("%Y-%m-%d %H-%M-%S")
        self.date = date
        self.resultPath = os.path.join(out_dir, "results-" + date)
        self.modelPath = f"models/{version}"
        os.makedirs(self.resultPath, exist_ok=True)
        os.makedirs(self.modelPath, exist_ok=True)


    def init_logger(self,):
        
        logger = logging.getLogger("multi_head_v1")
        logger.setLevel(logging.DEBUG)
        logpath = os.path.join(self.outDir, "multi_head_v1.log")
        fhandle = logging.FileHandler(logpath, mode="w")
        fhandle.setLevel(logging.DEBUG)
        
        f = logging.Formatter('FIRE DETECTOR | %(date)s | %(levelname)s | %(message)s')
        fhandle.setFormatter(f)
        logger.addHandler(fhandle)
        
        logger.info("Module is Initialized", extra={"date":self.date})

        self.logger = logger

        return True


    def setup_gpus(self,):

        """
        GPU adjusting function. If GPU's not available runs with cpu.
        
        """
        idx = self.gpus
        
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        
            print("GPU Properties: ", torch.cuda.get_device_properties(torch.cuda))
            print("Total number of GPUs: ", torch.cuda.device_count())
            print("Current Device", torch.cuda.current_device())
        
            torch.cuda.empty_cache()
            self.logger.info(f"Succesfully connected to the GPU: {idx}.", extra={"date":self.get_date()})
            print(f"Succesfully connected to the GPU: {idx}.")
    
        except Exception as e:
    
            self.logger.error(f"GPU's not available. Try to restrart docker.\nError:{e}", extra={"date":self.get_date()})
            print(f"GPU's not available. Try to restrart docker.\nError:{e}")


    def get_date(self,):
        gmt3 = timezone(timedelta(hours=3))
        dt = datetime.now(gmt3)
        dt = dt.strftime("%Y-%m-%d %H:%M:%S")
        return dt
        

    def read_jsonl(self, filepath):
        items = []
        try:
            with open(filepath, "r", encoding = "utf-8-sig") as f:
                for line in f:
                    item = json.loads(line)
                    items.append(item)
        except Exception as e:
            print(f"Error encountered while reading the jsonl.\nError:{e}")
            self.logger.error(f"Error encountered while reading the jsonl.\nError:{e}", extra={"date":self.get_date()})
            
        return items


    def write_jsonl(self, file, filepath, mode= "w"):
        try:
            with open(filepath, mode, encoding = "utf-8-sig") as f:
                for line in file:
                    item = json.dumps(line, ensure_ascii=False) + "\n"
                    f.write(item) 

        except Exception as e:
            print(f"Error encountered while writing to the jsonl.\nError:{e}")
            self.logger.error(f"Error encountered while writing to the jsonl.\nError:{e}", extra={"date":self.get_date()})
            return False          
            

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s.,\']', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    

    def loading_datasets(self, datapath=None, split=None):

        if datapath is None:
            datapath = self.dataPath
            
        if split is None:
            split = self.split

        if self.device == "cuda":
            self.setup_gpus()
            
        if os.path.exists(datapath):
            items = self.read_jsonl(datapath)
            if items:
                self.data = pd.DataFrame(items)
                print(f"Loaded Dataset FRom Local:\n\n{self.data.head()}")
                self.logger.info(f"Loaded Dataset FRom Local:\n\n{self.data.head()}", extra={"date":self.get_date()})
                return True

            else:
                print(f"The path: {datapath}, is not a jsonl path.")
                self.logger.error(f"The path: {datapath}, is not a jsonl path.", extra={"date":self.get_date()})
                try:
                    self.data = pd.read_csv(datapath)
                    print("The path corresponds a csv file")
                    self.logger.info("The path corresponds a csv file", extra={"date":self.get_date()})
                    return True

                except Exception as e:
                    self.data = []
                    print(f"It is not a csv path too.\nError:{e}")
                    self.logger.error(f"It is not a csv path too. Error: {e}")

                return False

        else:
            try:
                df = load_dataset(datapath, split=split)
                if df:
                    self.data = pd.DataFrame(df)
                    print(f"Loaded Dataset From Huggingface:\n\n{self.data.head()}")
                    self.logger.info(f"Loaded Dataset From Huggingface:\n\n{self.data.head()}", extra={"date":self.get_date()})
                    return True
                raise FileNotFoundError

            except:
                print(f"The path: {datapath}, seems to be wrong.")
                self.logger.error(f"The path: {datapath}, seems to be wrong.", extra={"date":self.get_date()})
                return False


    def preprocess_data(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batchSize

        b = batch_size
            
        FOLDER_PATH = "era5_monthly"
        i = 0
        X_l = []
        Y_l = []
        for dir in tqdm(os.listdir(FOLDER_PATH)):
            i+=1
            BASE_PATH = os.path.join(FOLDER_PATH, dir)

            try:

                x = np.load(os.path.join(BASE_PATH,"X.npy"), allow_pickle=True)
                y = np.load(os.path.join(BASE_PATH,"y.npy"), allow_pickle=True)
                X_l.append(x)
                Y_l.append(y)

            except:
                continue

        X = np.concatenate(X_l, axis=0)
        Y = np.concatenate(Y_l, axis=0)

        np.save("models/v1/data.npy", X)
        np.save("models/v1/label.npy", Y)

        print(f"Inputs and Labels are extracted. X shape: {X.shape}, Y shape: {Y.shape}")
        self.logger.info(f"Inputs and Labels are extracted. X shape: {X.shape}, Y shape: {Y.shape}", extra={"date":self.get_date()})

            
        N = len(X)

        train_end = int(N * 0.8)
        val_end   = int(N * 0.9)

        X_train = X[:train_end].copy()
        X_val   = X[train_end:val_end].copy()
        X_test  = X[val_end:].copy()
        Y_train = Y[:train_end].copy()
        Y_val = Y[train_end:val_end].copy()
        Y_test = Y[val_end:].copy()

        num_idx = [-3,-2,-1]

        mean = X_train[:, :, num_idx].mean(axis=(0, 1))
        std  = X_train[:, :, num_idx].std(axis=(0, 1))

        X_train[:, :, num_idx] = (X_train[:, :, num_idx] - mean) / std
        X_val[:, :, num_idx]   = (X_val[:, :, num_idx]   - mean) / std
        X_test[:, :, num_idx]  = (X_test[:, :, num_idx]  - mean) / std

        np.save("models/v1/num_mean.npy", mean)
        np.save("models/v1/num_std.npy", std)

        X_train = torch.tensor(X_train, dtype=torch.float32)

        X_val = torch.tensor(X_val, dtype=torch.float32)

        X_test = torch.tensor(X_test, dtype=torch.float32)

        Y_train = torch.tensor(Y_train, dtype=torch.float32)
        Y_val   = torch.tensor(Y_val, dtype=torch.float32)
        Y_test  = torch.tensor(Y_test, dtype=torch.float32)
        
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)
        val_dataset = TensorDataset(X_val, Y_val)
        # den_dataset = TensorDataset(den_ids, den_masks, den_labels)
        # d_dataset = TensorDataset(d_ids, d_masks, d_labels)
        
        train = DataLoader(train_dataset, batch_size=b, shuffle=True)
        test = DataLoader(test_dataset, batch_size=b)
        val = DataLoader(val_dataset, batch_size=b)
        # den = DataLoader(den_dataset, batch_size=b)
        # d = DataLoader(d_dataset, batch_size=b)

        self.trainData = train_dataset
        self.testData = test_dataset
        self.valData = val_dataset
        # self.denData = den_dataset
        # self.dData = d_dataset

        self.trainLoader = train
        self.testLoader = test
        self.valLoader = val
        # self.denLoader = den
        # self.dLoader = d

        print("Datasets and Dataloader are created successfully.")
        self.logger.info("Datasets and Dataloader are created successfully.", extra={"date":self.get_date()})
        return True
        

    def init_model(self, modelname=None):

        try:
                
            class LSTMFireModel(nn.Module):
                def __init__(self, input_dim=17, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
                    super().__init__()

                    self.lstm = nn.LSTM(
                        input_size=input_dim,      
                        hidden_size=hidden_dim,    
                        num_layers=num_layers,   
                        batch_first=True,          
                        dropout=dropout if num_layers > 1 else 0
                    )

                    self.fc = nn.Sequential(
                        nn.Linear(hidden_dim, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, output_dim) 
                    )

                def forward(self, x):

                    lstm_out, (h_n, c_n) = self.lstm(x)

                    last_hidden = h_n[-1]       

                    out = self.fc(last_hidden) 
                    return out

            self.model = LSTMFireModel()   

            print(f"{modelname}, was loaded successfully.")
            self.logger.info(f"{modelname}, was loaded successfully.", extra={"date":self.get_date()})

        except Exception as e:
            print(f"{modelname}, could not be loaded successfully. \nError:{e}")
            self.logger.error(f"{modelname}, could not be loaded successfully. \nError: {e}", extra={"date":self.get_date()})
            return False

        # self.model.classifier = nn.Sequential(
        #     nn.Linear(768, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64,5)
        # )
        # self.model.to(device)

        self.train_loss_store = []
        self.val_loss_store = []
        self.train_acc_store = []
        self.val_acc_store = []
        self.tr_cls_store = []
        self.v_cls_store = []

        return True

    def train_model(self, epochs=None, lr=None, wd=None):
        if epochs is None:
            epochs = self.epochs

        if lr is None:
            lr = self.lr

        if wd is None:
            wd = self.wd

        device = self.device
            
        model = self.model.to(device)
        train = self.trainLoader
        val = self.valLoader
        # train = self.denLoader
        # val = self.dLoader
        
        train_loss_storage =  self.train_loss_store
        val_loss_storage = self.val_loss_store
        
        optimizer = torch.optim.AdamW(params=model.parameters(), lr= lr, weight_decay = wd)
        pos_weight = torch.tensor([5346 / 1792])
        loss_mse = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        for epoch in tqdm(range(epochs)):
            model.train()
            train_loss_epoch = 0.0
            tr_reg_epoch = 0.0
            tr_reg_acc_epoch = 0.0
            i = 0
            for X_batch, y_batch in tqdm(train):
                i += 1
                X_batch = X_batch.to(device) ; y_reg = y_batch.type(dtype=torch.float32).to(device)
                reg_logits = model(X_batch).squeeze()
                
                probs = torch.sigmoid(reg_logits).cpu().detach().numpy()
                reg_pred = (probs > 0.7).astype(np.float32)
                reg_acc = (reg_pred == y_reg.cpu().detach().numpy()).astype(np.float32).mean()
                
                reg_loss = loss_mse(reg_logits, y_reg)
                
                optimizer.zero_grad()
                reg_loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss_epoch += reg_loss.item() * X_batch.size(0)  # sum of loss over batch
                tr_reg_acc_epoch += reg_acc * X_batch.size(0)
        
            avg_train_loss = train_loss_epoch / len(train.dataset)
            avg_tr_reg_acc = tr_reg_acc_epoch / len(train.dataset) 
        
            train_loss_storage.append(avg_train_loss)
        
            model.eval()
            val_loss_epoch = 0.0
            v_reg_epoch = 0.0
            v_reg_acc_epoch = 0.0
            prob_list = []
            with torch.no_grad():
                for X_batch, y_batch in val:
           
                    X_batch = X_batch.to(device) ; y_reg = y_batch.type(dtype=torch.float32).to(device)
                    reg_log = model(X_batch).squeeze()

                    v_prob = torch.sigmoid(reg_log).cpu().detach().numpy()
                    v_reg_pred = (v_prob > 0.7).astype(np.float32)
                    v_reg_acc = np.sum(v_reg_pred == y_reg.cpu().detach().numpy()).astype(np.float32).mean()
                    
                    v_loss = loss_mse(reg_log, y_reg)                 
                    
                    val_loss_epoch += v_loss.item() * X_batch.size(0)
                    v_reg_acc_epoch += v_reg_acc * X_batch.size(0) 
        
            avg_val_loss = val_loss_epoch / len(val.dataset)
            avg_v_reg_acc = v_reg_acc_epoch / len(val.dataset) 
            
            val_loss_storage.append(avg_val_loss)
                    
            print(f"""Epoch {epoch+1}/{epochs}\n\n - Train Loss: {avg_train_loss:.4f}\n - Train Accuracy: {avg_tr_reg_acc:.4f}\n\n - Val Loss: {avg_val_loss:.4f}\n - Val Accuracy: {avg_v_reg_acc:.4f}\n\n""")
            
            self.logger.info(f"""Epoch {epoch+1}/{epochs}\n\n - Train Loss: {avg_train_loss:.4f}\n - Train Accuracy: {avg_tr_reg_acc:.4f}\n\n - Val Loss: {avg_val_loss:.4f}\n - Val Accuracy: {avg_v_reg_acc:.4f}""", extra={"date":self.get_date()})

            if avg_val_loss <= self.bestLoss and avg_v_reg_acc >= self.bestAcc:
                self.bestLoss = avg_val_loss
                self.bestAcc = avg_v_reg_acc
                self.save_model()


        self.train_loss_store = train_loss_storage
        self.val_loss_store = val_loss_storage

        self.model = model
        return True
        

    def evaluate_model(self, result_path=None):
        if result_path is None:
            result_path = self.resultPath
        device = self.device
        model = self.model
        test = self.testLoader
        model = model.to(device)

        train_loss_storage = self.train_loss_store
        val_loss_storage = self.val_loss_store
        
        model.eval()
        loss_mse = nn.BCEWithLogitsLoss()
        
        test_loss_epoch = 0.0
        t_loss_epoch = 0.0
        t_reg_acc_epoch = 0.0
        prob_list = []
        pred_list = []
        test_labels = []
        reg_pre = []
        reg_labels = []
        with torch.inference_mode():
            for X_batch, y_batch in tqdm(test):
        
                    X_batch = X_batch.to(device) ; y_reg = y_batch.type(dtype=torch.float32).to(device)
                    reg_log = model(X_batch).squeeze()

                    probs = torch.sigmoid(reg_log).cpu().detach().numpy()
                    reg_pred = (probs > 0.7).astype(np.float32)
                    reg_acc = (reg_pred == y_reg.cpu().detach().numpy()).astype(np.float32).mean()
                
                    reg_pre += reg_pred.tolist()
                    reg_labels += y_reg.cpu().detach().tolist()

                    t_reg_loss = loss_mse(reg_log, y_reg)
                
                    t_loss_epoch += t_reg_loss.item() * X_batch.size(0)
                    t_reg_acc_epoch += reg_acc * X_batch.size(0) 
    
        avg_t_reg = t_loss_epoch / len(test.dataset)
        avg_t_reg_acc = t_reg_acc_epoch / len(test.dataset)
        
        reg_pre = np.array(reg_pre).astype(int)
        reg_labels = np.array(reg_labels).astype(int)

        print(f"""- Test Loss: {avg_t_reg:.4f}\n - Test Accuracy: {avg_t_reg_acc:.4f}\n\n""")
            
        self.logger.info(f"""- Test Loss: {avg_t_reg:.4f}\n - Test Accuracy: {t_reg_acc_epoch:.4f}\n\n""", extra={"date":self.get_date()})       

    
        # for i in range(4):
        if True:
            print(f"\n\nClassification Report for Regression")
            print(classification_report(reg_labels, reg_pre, target_names=np.unique(reg_labels).astype(str).tolist(), zero_division=0))

        # for i in range(4):
        if True:
            self.logger.info(f"\n\nClassification Report for Regression\n", extra={"date":self.get_date()})
            self.logger.info(classification_report(reg_labels, reg_pre, target_names=np.unique(reg_labels).astype(str).tolist(), zero_division=0), extra={"date":self.get_date()})
    

        # for i in range(4):
        if True:
            cm = confusion_matrix(reg_labels, reg_pre)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(reg_labels).astype(str).tolist())
                
            # CONFUSION MATRIX
            plt.figure(figsize=(10, 8))
            disp.plot(xticks_rotation="vertical", cmap="Blues", values_format="d")
            plt.title(f"Confusion Matrix")
            cmsave = os.path.join(result_path, f"confusion_matrix.png")
            plt.savefig(cmsave)
            plt.close()

        # TRAIN FUNCTIONS
        plt.figure()
        plt.plot(range(1, len(train_loss_storage) + 1), train_loss_storage, color="red", label="Train Loss")
        plt.plot(range(1, len(val_loss_storage) + 1), val_loss_storage, color="blue", label="Validation Loss")
        plt.legend()
        trsave = os.path.join(result_path, "train_loss_functions.png")
        plt.savefig(trsave)
        plt.close()


        # # PREDICTION LINES
        # plt.figure()
        # plt.plot(range(1, len(test_loss_storage) + 1), test_loss_storage, color="blue", label="Test Loss")
        # plt.legend()
        # psave = os.path.join(result_path, "prediction_loss_functions.png")
        # plt.savefig(psave)
        # plt.close()        

        print(f"Plots saved to the path: {result_path}. Evaluation is finished successfully.")
        self.logger.info(f"Plots saved to the path: {result_path}. Evaluation is finished successfully.", extra={"date":self.get_date()})
        return True


    def save_model(self, modelpath = None, loss=None, acc=None):
        if modelpath is None:
            modelpath = self.modelPath

        if loss is None:
            loss = self.bestLoss

        if acc is None:
            acc = self.bestAcc
        fold = Path(modelpath)
        fold.mkdir(exist_ok=True)
        name = f"model_bert_loss_{loss:.2f}_acc_{acc:.2f}.pth"
        file = os.path.join(fold, name)
        torch.save(self.model.state_dict(), file)
        return True

    def load_model(self, savepath=None):
        if not savepath:
            return False


        class LoadModel(nn.Module):
                def __init__(self,):
                    super().__init__()
                    #self.model = AutoModel.from_pretrained(modelname)
                    self.multi_head = nn.Sequential(
                        nn.BatchNorm1d(768),
                        nn.Linear(768, 64),
                        nn.BatchNorm1d(64),
                        nn.Dropout(p=0.1),
                        nn.ReLU(),
                        nn.Linear(64,4)
                    )

                def forward(self,x, attention_mask):
                    out = self.bert(x, attention_mask=attention_mask)
                    out = out.last_hidden_state[:, 0, :]
                    out2 = self.multi_head(out)
                    return out2

        model = LoadModel()
        model.load_state_dict(torch.load(savepath))
        self.model = model
        for param in self.model.bert.parameters():
                param.requires_grad = False

        for param in self.model.bert.embeddings.parameters():
                param.requires_grad = True


        for param in self.model.bert.transformer.layer[-1].parameters():
                    param.requires_grad = True

        self.train_loss_store = []
        self.val_loss_store = []
        self.tr_reg_store = []
        self.v_reg_store = []
        return True
        

    def save(self, x, model=None, embed=None, b=None, result_path=None):
        if model is None:
            model = self.model

        if embed is None:
            embed = self.tokenizer

        if b is None:
            b = self.batchSize

        if result_path is None:
            result_path = self.resultPath

        device = self.device
            
        model = model.to(device)
        model.eval()

        X = [self.clean_text(para) for para in x]        
        try:
            I = embed(X, padding=True, truncation=True, return_tensors="pt")
            print(f"Inputs was tokenized successfully.")
            self.logger.error(f"Inputs was tokenized successfully.", extra={"date":self.get_date()})
        except Exception as e:
            print(f"Inputs could not be tokenized.\n{e}")
            self.logger.error(f"Inputs could not be tokenized.\n{e}", extra={"date":self.get_date()})
            return False
        data = TensorDataset(I["input_ids"], I["attention_mask"])
        X_loader = DataLoader(data, batch_size=b)

        prob_list = []
        pred_list = []
        with torch.no_grad():
            for X_batch, mask in X_loader:
                    mask = (mask == 0)
                    X_batch = X_batch.to(device) ; mask = mask.to(device)
                    log = model(X_batch, mask).squeeze()
                    prob_list += log.cpu().detach().tolist()
                    pred =  torch.round(log)
                    pred_list += pred.cpu().detach().tolist()
    
        probs = np.array(prob_list)
        print("Probabilities (min, max, mean):", probs.min(), probs.max(), probs.mean())
        self.logger.info(f"Probabilities (min, max, mean): {probs.min()}, {probs.max()}, {probs.mean()}", extra={"date":self.get_date()})     

        result = [{"Input":i, "Score":p} for i, p in list(zip(x,pred_list))]
        self.write_jsonl(result, result_path)    
        
        return True
        

    def main(self,):
        while True:
            self.init_logger()
            
            load_status = self.loading_datasets(self.dataPath, self.split)  # self.data is gathered
            if not load_status:
                print("The Data could not be loaded successfully. Exiting...")
                self.logger.error("The Data could not be loaded successfully. Exiting...", extra={"date":self.get_date()})
                break
                
            pre_status = self.preprocess_data(self.embed, self.batchSize)  # data is passed as self.data, dataloaders gathered.
            if not pre_status:
                print("The Data could not be processed successfully.")
                self.logger.error("The Data could not be processed successfully.", extra={"date":self.get_date()})
                break
            return
                
            init_status = self.init_model(self.modelName)
            if not init_status:
                print("The Model could not be initialized successfully.")
                self.logger.error("The Model could not be initialized successfully.", extra={"date":self.get_date()})
                break
            
            train_status = self.train_model(self.epochs, self.lr, self.wd)
            if not train_status:
                print("The Model could not be trained successfully.")
                self.logger.error("The Model could not be trained successfully.", extra={"date":self.get_date()})
                break
                
            eval_status = self.evaluate_model(self.resultPath)
            if not eval_status:
                print("The Outcomes could not be evaluated successfully.")
                self.logger.error("The Outcomes could not be evaluated successfully.", extra={"date":self.get_date()})
                break

            break
            
        # save_status = self.save()
        # if not save_status:
        #     print("The Outcomes could not be saved successfully.")
        #     logger.error("The Outcomes could not be saved successfully.")
        #     break
        
        
        
        








