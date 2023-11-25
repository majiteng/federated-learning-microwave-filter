import pickle
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

fn_001 = './save/results_rho_0.01_noise_balanced_share.pkl'
fn_005 = './save/results_rho_0.05_noise_balanced_share.pkl'
fn_010 = './save/results_rho_0.10_noise_balanced_share.pkl'
fn_base = './save/results_base.pkl'

for fn in [fn_base, fn_001, fn_005, fn_010]:
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        pred = data['pred']
        truth = data['truth']
        list_r2 = []
        for i in range(truth.shape[1]):
            r2 = metrics.r2_score(truth[:, i].numpy().ravel(), pred[:, i].numpy().ravel())
            list_r2.append(r2)
            # print('Col {:}, R2 {:.4f}'.format(i, r2))
        print('Avg R2: {:.4f}'.format(sum(list_r2) / len(list_r2)))

file_name = './save/loss_acc.pkl'
loss_001 = './save/loss_acc_rho_0.01_noise_balanced_share.pkl'
loss_005 = './save/loss_acc_rho_0.05_noise_balanced_share.pkl'
loss_010 = './save/loss_acc_rho_0.10_noise_balanced_share.pkl'
loss_base = './save/loss_acc_base.pkl'

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for marker, rho, loss in zip(['o', 's', '*', 'd'], [0.0, 0.01, 0.05, 0.10], [loss_base, loss_001, loss_005, loss_010]):
    with open(loss, 'rb') as f:
        data = pickle.load(f)
        train_loss = data['train_loss']
        val_loss = data['val_loss']
        train_r2 = data['train_r2']
        val_r2 = data['val_r2']

        axes[0, 0].plot(train_loss, marker, linestyle='-', markevery=10, markerfacecolor='white', label=r'Train Loss with $\rho={:.2f}$'.format(rho))
        axes[0, 1].plot(val_loss, marker, linestyle='-', markevery=10, markerfacecolor='white', label=r'Val Loss with $\rho={:.2f}$'.format(rho))
        axes[1, 0].plot(train_r2, marker, linestyle='-', markevery=10, markerfacecolor='white', label=r'Train R2 with $\rho={:.2f}$'.format(rho))
        axes[1, 1].plot(val_r2, marker, linestyle='-', markevery=10, markerfacecolor='white', label=r'Val R2 with $\rho={:.2f}$'.format(rho))

        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss (MSE)')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('R2 Score')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('R2 Score')

        axes[0, 0].legend()
        axes[0, 1].legend()
        axes[1, 0].legend()
        axes[1, 1].legend()

plt.tight_layout()
plt.savefig('./save/fig1.pdf', dpi=300)
plt.show()
