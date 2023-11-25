import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP1
from models import MLP2
from utils import get_dataset, average_weights, exp_details
from sklearn import preprocessing

import csv 
import pandas as pd
import codecs

def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("save the csv file")

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('./logs')

    args = args_parser()
    exp_details(args)

    # device = 'cuda' if args.gpu else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
    else:
      device = torch.device("cpu")
      print("PyTorch is running on CPU.")
      
    # load dataset and user groups
    train_dataset, test_dataset, scale_a=get_dataset()

    # BUILD MODEL
    #global_model = MLP1(dim_in=12, dim_hidden=1024, dim_out=32)
    global_model = MLP2()

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
#     print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_r2 = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    val_acc = []
    val_r2 = []
    val_loss = []
  
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_r2 = []
        # print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = ['A', 'B']

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset[idx], logger=logger)
            w, loss, r2 = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_r2.append(r2)
       
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        r2_avg = sum(local_r2) / len(local_r2)
        
        train_loss.append(loss_avg)
        train_r2.append(r2_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss, list_r2 = [], [], []
        global_model.eval()
        for client in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset[client], logger=logger)
            acc, loss, r2 = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
            list_r2.append(r2)
        val_r2_avg = sum(list_r2) / len(list_r2)
        val_acc.append(sum(list_acc) / len(list_acc))
        val_r2.append(val_r2_avg)
        val_loss.append(sum(list_loss) / len(list_loss))
        
        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print('\nRound: {:}, Loss: {:.6f}, Train R2: {:6f}, Val R2: {:.6f}, '
                  .format(epoch + 1, sum(train_loss) / len(train_loss), r2_avg, val_r2_avg))
            
    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    # Test inference after completion of training

    # Test on client A data
#     test_acc, test_loss, final_preds, final_truth,fianl_PREall,fianl_TRUall,final_predsA,final_truthA,testA3= test_inference(global_model, test_dataset)
    
#     print('Results after {:} epochs: Train Acc. {:.2f}%, Test Acc. {:.2f}%'.format(args.epochs, 100 * val_acc[-1],
#                                                                                    100 * test_acc))
   
   #  print(testA3[0])
   #  print(fianl_PREall[0])
   #  print(fianl_TRUall[0])
    
   # # final_preds1=final_preds[:int(len(final_preds) / 2)].detach().numpy()
   # # final_truth1=final_truth[:int(len(final_truth) / 2)].detach().numpy()

   #  final_preds111=scale_aT.inverse_transform(fianl_PREall)
   #  final_truth111=scale_aT.inverse_transform(fianl_TRUall)
    

   #  print(final_preds111[0])
   #  print(final_truth111[0])

     
    
    #print(final_preds_print)
    #print(final_truth_print)
    
    # Saving the objects train_loss and train_accuracy:
#     file_name = './save/loss_acc.pkl'
#     with open(file_name, 'wb') as f:
#         # pickle.dump([train_loss, train_r2, val_acc, val_loss, val_r2], f)
#         pickle.dump(obj={'train_loss': train_loss,
#                          'train_r2': train_r2,
#                          'val_acc': val_acc,
#                          'val_loss': val_loss,
#                          'val_r2': val_r2}, file=f)
#     f.close()

#     save_name = './save/results.pkl'
#     with open(save_name, 'wb') as f:
#         pickle.dump(obj={'pred': final_preds,
#                          'truth': final_truth}, file=f)
#     f.close()

    
   #print(val_loss)
#     val_loss_csv = pd.DataFrame(val_loss)
#     val_loss_csv.to_csv('./val_loss_MLINE464-0.1.csv')
#     print(val_r2)
#     val_r2_csv = pd.DataFrame(val_r2)
#     val_r2_csv.to_csv('./val_r2_MLINE464—0.1.csv')
#     print(batch_r2_sum)
#     batch_r2_csv = pd.DataFrame(batch_r2_sum)
#     batch_r2_csv.to_csv('./batch_r2_sum.csv')