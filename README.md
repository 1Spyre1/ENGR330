ble.py --> To connect HM-10 BLE module and save recordings to the data.txt file

cds_fire_merge_all.py --> To merge monthly downloaded ERA5 files to a big DataFrame. Then, the DataFrame will be preprocessed to create torch tensors to used directly to train model.

data.txt --> recordings of BLE module. It is read by UI.

model_bert_loss_0.43_acc_8117.00.pth --> Contains the weights of best performed LSTM model.

main.c --> MSP430FR5969 implementation with the BME280 and HC-05 Bluetooth Module. It consists of I2C protocole implementation and also energy efficiency implementations.

main.ino --> Arduino code for our system. When MCU burned, we transferred to Arduino.

main.py --> Contains the AI training class which is FirePredictionModule. It consists of training, saving and evaluation parts as a whole system.

ui.py --> main code for Streamlit UI.

utils.py --> Utils funcitons for general purposes.

v1.ipynb --> Wildfire data analysis and FirePredictionModule implementation to train model. 
