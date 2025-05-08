import torch
import numpy as np
from tqdm import tqdm
from src.models.Transformers import _generate_square_subsequent_mask
from torch.utils.data import DataLoader

def clas(tensor):
    for i in range(len(tensor)):
        if tensor[i]<=0:
            tensor[i]=0
        else:
            tensor[i]=1
    return tensor

def evaluate(model, criterion, dataloader, tgt_mask, pbar, device='cpu', dataset_name='train'):
    model.eval()
    total_loss = 0
    class_loss = 0
    regress_loss = 0
    n = 0
    n_fix, n_blink = 0, 0
    correct_fix, correct_blink = 0, 0
    durations_fix, durations_blink = [], []
    pbar.set_description(f"Evaluating {dataset_name} dataset on {device}")
    for tgt, tgt_y in dataloader:
        if tgt.size(0)!=tgt_mask.size(0):
                tgt_mask = _generate_square_subsequent_mask(tgt.size(1), device=device).repeat(tgt.size(0)*model.nhead, 1, 1)
        n+=tgt.size(0)
        # src = src.permute(1, 0, 2)
        # tgt = tgt.permute(1, 0, 2)

        with torch.no_grad():
        # Make forecasts
            classes_prediction, duration_prediction = model(tgt=tgt, tgt_mask=None)#.squeeze()#.permute(1, 0, 2).squeeze()
            # print(tgt.shape, tgt_y.shape)
            # print(classes_prediction.shape, duration_prediction.shape)
            # raise("Hola")
        
            fix_ind = tgt_y[:,-1,0]==0
            blink_ind = tgt_y[:,-1,0]==1
            n_fix+=len(fix_ind)
            n_blink+=len(blink_ind)
            correct_fix+=int((clas(classes_prediction[fix_ind,-1].cpu().argmax(dim=1))==clas(tgt_y[fix_ind,-1,0].cpu())).sum())
            correct_blink+=int((clas(classes_prediction[blink_ind,-1].cpu().argmax(dim=1))==clas(tgt_y[blink_ind,-1,0].cpu())).sum())
            # correct+=int((clas(classes_prediction[:,-1].cpu().argmax(dim=1))==clas(tgt_y[:,-1,0].cpu())).sum())
            # Calculate loss
            losses = criterion(classes_prediction, duration_prediction, tgt_y)#.squeeze())
            class_loss+=losses["class"]
            regress_loss+=losses["regress"]
            total_loss+=sum(losses.values())
            # print(duration_prediction.shape, tgt_y.shape)
            for duration_pred, duration_real, c in zip(duration_prediction[:,-1,0].cpu(), tgt_y[:,-1,1].cpu(), tgt_y[:,-1,0].cpu()):
                if int(c)==0:
                    durations_fix.append(4000*(duration_pred-duration_real))
                else:
                    durations_blink.append(4000*(duration_pred-duration_real))
        pbar.update()
        metrics = {"Loss/total/"+dataset_name: total_loss/n, "Loss/regress/"+dataset_name: regress_loss/n, "Loss/class/"+dataset_name: class_loss/n,
                   "Acc/total/"+dataset_name: (correct_fix+correct_blink)/n,
                   "Mean_delay/total/"+dataset_name: np.mean(durations_fix+durations_blink),
                   "Std_delay/total/"+dataset_name: np.std(durations_fix+durations_blink),
                   "Delay/total/"+dataset_name: np.array(durations_fix+durations_blink)}
        
        if n_fix>0:
            metrics["Acc/fix/"+dataset_name] = correct_fix/n_fix
        if n_blink>0:
            metrics["Acc/blink/"+dataset_name] = correct_blink/n_blink

        if len(durations_fix)>0:
            metrics["Mean_delay/fix/"+dataset_name] = np.mean(durations_fix)
            metrics["Std_delay/fix/"+dataset_name] = np.std(durations_fix)
            metrics["Delay/fix/"+dataset_name] = np.array(durations_fix)
            
        if len(durations_blink)>0:
            metrics["Mean_delay/blink/"+dataset_name] = np.mean(durations_blink)
            metrics["Std_delay/blink/"+dataset_name] = np.std(durations_blink)
            metrics["Delay/blink/"+dataset_name] = np.array(durations_blink)
    
    
    return metrics

def get_performance(model, criterion, dataset, tgt_mask, device='cpu', dataset_name='train'):
    # dataloader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=1)#, pin_memory=True)
    
    model.eval()
    total_loss = []
    class_loss = []
    regress_loss = []
    n = 0
    n_fix, n_blink = 0, 0
    correct_fix, correct_blink = 0, 0
    durations, durations_fix, durations_blink = [], [], []
    pbar = tqdm(range(len(dataset)), leave=False)
    pbar.set_description(f"Evaluating {dataset_name} dataset on {device}")

    for tgt, tgt_y in dataset:
        tgt, tgt_y = tgt.unsqueeze(0), tgt_y.unsqueeze(0)
        if tgt.size(0)!=tgt_mask.size(0):
                tgt_mask = _generate_square_subsequent_mask(tgt.size(1), device=device).repeat(tgt.size(0)*model.nhead, 1, 1)
        n+=1

        with torch.no_grad():
            classes_prediction, duration_prediction = model(tgt=tgt, tgt_mask=None)
            
            losses = criterion(classes_prediction, duration_prediction, tgt_y)
            class_loss.append(float(losses["class"]))
            regress_loss.append(float(losses["regress"]))
            total_loss.append(float(losses["class"])+float(losses["regress"]))

            duration_pred, duration_real = duration_prediction[0,-1,0].cpu(), tgt_y[0,-1,1].cpu()

            if int(classes_prediction[0, -1, 0].cpu()) == int(tgt_y[0, -1, 0].cpu()):
                correct = 1
            else:
                correct = 0

            if int(tgt_y[0, -1, 0].cpu()) == 0:
                n_fix+=1
                correct_fix+=correct
                durations_fix.append(4000*(duration_pred-duration_real))
            else:
                n_blink+=1
                correct_blink+=correct
                durations_blink.append(4000*(duration_pred-duration_real))
            durations.append(4000*(duration_pred-duration_real))

        pbar.update()
        metrics = {"Loss/total/": total_loss, "Loss/regress/": regress_loss, "Loss/class/": class_loss,
                   "Mean_Loss/total/": sum(total_loss)/n, "Mean_Loss/regress/": sum(regress_loss)/n, "Mean_Loss/class/": sum(class_loss)/n,
                   "Acc/total/": (correct_fix+correct_blink)/n,
                   "Mean_delay/total/": np.mean(durations_fix+durations_blink),
                   "Std_delay/total/": np.std(durations_fix+durations_blink),
                   "Delay/total/": np.array(durations)}
        
        if n_fix>0:
            metrics["Acc/fix/"] = correct_fix/n_fix
        if n_blink>0:
            metrics["Acc/blink/"] = correct_blink/n_blink

        if len(durations_fix)>0:
            metrics["Mean_delay/fix/"] = np.mean(durations_fix)
            metrics["Std_delay/fix/"] = np.std(durations_fix)
            metrics["Delay/fix/"] = np.array(durations_fix)
            
        if len(durations_blink)>0:
            metrics["Mean_delay/blink/"] = np.mean(durations_blink)
            metrics["Std_delay/blink/"] = np.std(durations_blink)
            metrics["Delay/blink/"] = np.array(durations_blink)
    
    
    return metrics