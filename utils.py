from torch import nn
import torch
import numpy as np
import re
import pandas as pd

class LSTMFireModel(nn.Module):
                def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
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
                

def loading_fire_prediction_model():
    model = LSTMFireModel()
    model.load_state_dict(torch.load("models/fireprediction.pth"))
    print(model)
    return model


def predict_fire(model, X):
    with torch.inference_mode():
        reg_log = model(X).squeeze()
        probs = torch.sigmoid(reg_log).cpu().detach().numpy()
        reg_pred = (probs > 0.5).astype(np.float32)

    return reg_pred, probs


def read_txt(dataPath):
    readings = []
    reads = {}
    pattern = r"T:([\d.]+),H:([\d.]+),P:([\d.]+)"
    s_count = 0
    t_count = 0
    with open(dataPath, "r", encoding="utf-8") as f:
        start_reading = True
        for line in f:
            print(line)
            line = line.strip()
            if start_reading:
                if not line.startswith("T:"):    
                    continue
                start_reading = False

            if line.startswith("T:"):
                reads = {}
                match = re.search(pattern, line)
                if match:
                        # Extracting the groups
                        reads["sample_id"] = s_count
                        reads["time_id"] = t_count
                        reads["temperature"] = match.group(1)
                        reads["humidity"]  = match.group(2)
                        reads["pres"] = match.group(3)

            elif line[0].isdigit():
                reads["pres"] += line
                readings.append(reads)

    pd.DataFrame(readings)

    return readings
                
                
            
