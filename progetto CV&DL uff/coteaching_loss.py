import torch 
import torch.nn.functional as F
import numpy as np

# Loss functions che prende come argomenti le predizioni dei due modelli, il tasso di dimenticanza e gli indici del batch
def loss_coteaching(y_1, y_2, t, forget_rate, ind):
    t= t.float()

    loss_1 = F.binary_cross_entropy(y_1, t, reduction='none') # calcolo la loss per il primo modello
    ind_1_sorted = torch.argsort(loss_1.detach().view(-1)).cpu()  # Sposta su CPU prima di convertire in numpy
    loss_1_sorted = loss_1[ind_1_sorted] # viene ordinata la loss

    loss_2 = F.binary_cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = torch.argsort(loss_2.detach().view(-1)).cpu()  # Sposta su CPU prima di convertire in numpy
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    print(f'forget_rate: {forget_rate}')
    print(f'remember_rate: {remember_rate}')
    num_remember = int(remember_rate * len(loss_1_sorted)) # si calcola quanti esempi ricordare
    print(f'num_elementi: {len(loss_1_sorted)}')
    print(f'num_remember: {num_remember}')

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember] # indici da ricordare

    return ind_1_update, ind_2_update 
