import numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle, torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from src.evaluate import evaluate
from src.utils import get_empty_dict, split_dataset
from src.models.Transformers import _generate_square_subsequent_mask

from src.metrics.tensorboard import add_metrics_to_writer

def train(model,
    train_dataset,
    validation_dataset,
    tgt_mask,
    criterion,
    optimizer,
    device = 'cpu',
    scheduler = None,
    early_stop = None,
    epochs: int=5,
    batch_size: int=1,
    save_checkpoint: bool=True,
    dir_checkpoint = "",
    description = ""):

    if not dir_checkpoint.endswith("/"):
        dir_checkpoint = dir_checkpoint + "/"

    writer = SummaryWriter(dir_checkpoint)

    with open(str(dir_checkpoint)+"model_&_train_params.pkl", "wb") as f:
        model_params = {"d_model":model.d_model, "nhead": model.nhead,
                        "batch_first": model.batch_first, "num_decoder_layers": model.num_decoder_layers, 
                        "num_classes": model.num_classes, "dim_feedforward": model.dim_feedforward,
                        "dropout": model.dropout, "activation": model.activation.__name__,
                        "seq_len": model.seq_len}
        train_params = {"epochs": epochs, "lr": optimizer.param_groups[0]["lr"], "optimizer": str(optimizer),
                        "early_stop": early_stop, "batch_size": batch_size, "scheduler": scheduler_to_str(scheduler),
                        "criterior": str(criterion), "description": description}
                    
        pickle.dump({"model_params": model_params, "train_params": train_params}, f)
    # dir_checkpoint = dir_checkpoint+time_path(name_dir)+'/'

    # train_params = {"model": model,
    #                 "criterion": criterion,
    #                 "optimizer": optimizer,
    #                 "device": device,
    #                 "classes_labels": classes_labels,
    #                 "scheduler": scheduler,
    #                 "early_stop": early_stop,
    #                 "epochs": epochs,
    #                 "batch_size": batch_size,
    #                 "val_percent": val_percent,
    #                 "save_checkpoint": save_checkpoint,
    #                 "dir_checkpoints": dir_checkpoint,
    #                 "name_dir": name_dir}
    # Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    # with open(str(dir_checkpoint)+"train_parameters.pkl", "wb") as f:
    #         pickle.dump(train_params, f)

    # metrics = {"train_loss": [], "validation_loss": [], "train_acc": [], "validation_acc": [], "train_dur_dif_mean": [], "validation_dur_dif_mean": [], "train_dur_dif_std": [], "validation_dur_dif_std": []}

    validation_losses = []
    # # 1. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 2. Create data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=batch_size)#, pin_memory=True)
    val_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, batch_size=batch_size)#, pin_memory=True)
    i=0

    pbar = tqdm(total=2*len(train_loader)+len(val_loader),unit="batch", colour='blue', leave=False)
    metrics = evaluate(model, criterion, train_loader, tgt_mask, pbar=pbar, device=device, dataset_name='Train')
    add_metrics_to_writer(metrics, 0, writer)
    metrics = evaluate(model, criterion, val_loader, tgt_mask, pbar=pbar, device=device, dataset_name='Val')
    add_metrics_to_writer(metrics, 0, writer)

    # 3. Begin training
    for epoch in range(1,epochs+1):
        model.train()
        pbar = tqdm(total=2*len(train_loader)+len(val_loader),unit="batch", colour='blue', leave=False)
        pbar.set_description(f"Training on {device} - Epoch {epoch}/{epochs}")
        batch_loss = 0
        n = 0
        for (tgt, tgt_y) in train_loader:
            i+=1
            if model.nhead*tgt.size(0)!=tgt_mask.size(0):
                # print("si", tgt.shape, tgt_mask.shape)
                tgt_mask = _generate_square_subsequent_mask(tgt.size(1), device=device).repeat(tgt.size(0)*model.nhead, 1, 1)
            n+=tgt.size(0)
            # src = src.permute(1, 0, 2).to(device)
            # tgt = tgt.permute(1, 0, 2).to(device)

            # Make forecasts
            # print(tgt.shape, tgt_mask.shape)
            classes_prediction, duration_prediction = model(tgt=tgt, tgt_mask=tgt_mask)#.permute(1, 0, 2).squeeze()
            # print(tgt.shape, tgt_y.shape)
            # print(classes_prediction.shape, duration_prediction.shape)
            # raise("Hola")

            # prediction[:,:-1,:] = tgt_y[:,1:,:]

            # Zero the parameter gradients
            optimizer.zero_grad()
            # print(prediction.squeeze(), tgt_y.squeeze())
            # Calculate loss
            loss = criterion(classes_prediction, duration_prediction, tgt_y, reduct=True)
            # print(prediction, tgt_y)
            loss.backward() # Backpropogate loss

            optimizer.step() # Apply gradient descent change to weight
            batch_loss+=loss.item()

            # writer.add_scalar("Loss/Train_batch",loss.item(), i)
            pbar.set_postfix(**{"Batches mean loss": batch_loss/n})
            pbar.update()
        
        #AGREGAR 
        metrics = evaluate(model, criterion, train_loader, tgt_mask, pbar=pbar, device=device, dataset_name='Train')
        add_metrics_to_writer(metrics, epoch, writer)

        # metrics["train_loss"].append(train_loss)
        # metrics["train_acc"].append(train_acc)
        # metrics["train_dur_dif_mean"].append(train_dur_dif_mean)
        # metrics["train_dur_dif_std"].append(train_dur_dif_std)

        metrics = evaluate(model, criterion, val_loader, tgt_mask, pbar=pbar, device=device, dataset_name='Val')
        validation_losses.append(metrics["Loss/regress/Val"])
        add_metrics_to_writer(metrics, epoch, writer)
        
        # metrics["validation_loss"].append(validation_loss)
        # metrics["validation_acc"].append(validation_acc)
        # metrics["validation_dur_dif_mean"].append(validation_dur_dif_mean)
        # metrics["validation_dur_dif_std"].append(validation_dur_dif_std)
        # pbar.colour='green'
        # pbar.set_description(f"Epoch {epoch}/{epochs}")
        # pbar.set_postfix(**{"Train loss": train_loss, "Validation loss": validation_loss, "Validation delta duration": validation_dur_dif_mean, "±": validation_dur_dif_std})
        # pbar.bar_format='{desc}{bar}{percentage:3.0f}% | Total time: {elapsed}{postfix}'
        # pbar.refresh()
        # metrics["time"]+=pbar.format_dict["elapsed"]
        # merge_dict(metrics,evaluate(model, criterion, train_loader, device, dataset_name="train"))
        # merge_dict(metrics,evaluate(model, criterion, val_loader, device, dataset_name="val"))

        if scheduler:
            scheduler.step(metrics["Loss/regress/Val"])

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint+"last_model.pth"))
            if min(validation_losses)==validation_losses[-1]:
                torch.save(model.state_dict(), str(dir_checkpoint+"best_model.pth"))
        
        if early_stop:
            if epoch>early_stop:
                if min(validation_losses[-early_stop:]) >= min(validation_losses[:-early_stop]):
                    print(f"Early stop on epoch {epoch} (of {epochs})")
                    break

def train_final_layers(model,
    train_dataset,
    validation_dataset,
    tgt_mask,
    criterion,
    optimizer_regress,
    optimizer_class,
    device = 'cpu',
    scheduler = None,
    early_stop = None,
    epochs: int=5,
    batch_size: int=1,
    save_checkpoint: bool=True,
    dir_checkpoint = "",
    description = ""):

    if not dir_checkpoint.endswith("/"):
        dir_checkpoint = dir_checkpoint + "/"

    writer = SummaryWriter(dir_checkpoint)

    with open(str(dir_checkpoint)+"model_&_train_params.pkl", "wb") as f:
        model_params = {"d_model":model.d_model, "nhead": model.nhead,
                        "batch_first": model.batch_first, "num_decoder_layers": model.num_decoder_layers, 
                        "num_classes": model.num_classes, "dim_feedforward": model.dim_feedforward,
                        "dropout": model.dropout, "activation": model.activation.__name__,
                        "seq_len": model.seq_len}
        train_params = {"epochs": epochs, "lr_regress": optimizer_regress.param_groups[0]["lr"], "optimizer_regress": str(optimizer_regress),
                        "lr_class": optimizer_class.param_groups[0]["lr"], "optimizer_class": str(optimizer_class),
                        "early_stop": early_stop, "batch_size": batch_size, "scheduler": scheduler_to_str(scheduler),
                        "criterior": str(criterion), "description": description}
                    
        pickle.dump({"model_params": model_params, "train_params": train_params}, f)

    validation_losses_class, validation_losses_regress = [], []
    
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=batch_size)#, pin_memory=True)
    val_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, batch_size=batch_size)#, pin_memory=True)
    i=0

    pbar = tqdm(total=2*len(train_loader)+len(val_loader),unit="batch", colour='blue', leave=False)
    metrics = evaluate(model, criterion, train_loader, tgt_mask, pbar=pbar, device=device, dataset_name='Train')
    add_metrics_to_writer(metrics, 0, writer)
    metrics = evaluate(model, criterion, val_loader, tgt_mask, pbar=pbar, device=device, dataset_name='Val')
    add_metrics_to_writer(metrics, 0, writer)

    early_stop_regress, early_stop_class = False, False

    # 3. Begin training
    for epoch in range(1,epochs+1):
        model.train()
        pbar = tqdm(total=2*len(train_loader)+len(val_loader),unit="batch", colour='blue', leave=False)
        pbar.set_description(f"Training on {device} - Epoch {epoch}/{epochs}")
        batch_loss = 0
        n = 0
        for (tgt, tgt_y) in train_loader:
            i+=1
            if model.nhead*tgt.size(0)!=tgt_mask.size(0):
                # print("si", tgt.shape, tgt_mask.shape)
                tgt_mask = _generate_square_subsequent_mask(tgt.size(1), device=device).repeat(tgt.size(0)*model.nhead, 1, 1)
            n+=tgt.size(0)

            # Make forecasts
            classes_prediction, duration_prediction = model(tgt=tgt, tgt_mask=tgt_mask)#.permute(1, 0, 2).squeeze()

            # prediction[:,:-1,:] = tgt_y[:,1:,:]

            # Zero the parameter gradients
            optimizer_regress.zero_grad()
            optimizer_class.zero_grad()
            # Calculate loss
            loss = criterion(classes_prediction, duration_prediction, tgt_y, reduct=False)
            # print(prediction, tgt_y)
            if not early_stop_regress:
                loss["regress"].backward() # Backpropogate loss
            if not early_stop_class:
                loss["class"].backward() # Backpropogate loss

            optimizer_regress.step() # Apply gradient descent change to weight
            optimizer_class.step()
            batch_loss+=sum(loss.values()).item()

            # writer.add_scalar("Loss/Train_batch",loss.item(), i)
            pbar.set_postfix(**{"Batches mean loss": batch_loss/n})
            pbar.update()
        
        #AGREGAR 
        metrics = evaluate(model, criterion, train_loader, tgt_mask, pbar=pbar, device=device, dataset_name='Train')

        if not early_stop_regress:
            metrics_regress = metrics.copy()
            for key in metrics.keys():
                if key in ["Loss/total/Train", "Loss/class/Train", "Acc/total/Train", "Acc/fix/Train", "Acc/blink/Train"]:
                    metrics_regress.pop(key)
            add_metrics_to_writer(metrics_regress, epoch, writer)

        if not early_stop_class:
            metrics_class = metrics.copy()
            for key in metrics.keys():
                if key in ["Loss/regress/Train", "Mean_delay/total/Train", "Std_delay/total/Train", "Delay/total/Train", "Mean_delay/fix/Train", "Std_delay/fix/Train", "Delay/fix/Train", "Mean_delay/blink/Train", "Std_delay/blink/Train", "Delay/blink/Train"]:
                    metrics_class.pop(key)
            add_metrics_to_writer(metrics_class, epoch, writer)

        metrics = evaluate(model, criterion, val_loader, tgt_mask, pbar=pbar, device=device, dataset_name='Val')

        if not early_stop_regress:
            metrics_regress = metrics.copy()
            for key in metrics.keys():
                if key in ["Loss/total/Val", "Loss/class/Val", "Acc/total/Val", "Acc/fix/Val", "Acc/blink/Val"]:
                    metrics_regress.pop(key)
            add_metrics_to_writer(metrics_regress, epoch, writer)

        if not early_stop_class:
            metrics_class = metrics.copy()
            for key in metrics.keys():
                if key in ["Loss/regress/Val", "Mean_delay/total/Val", "Std_delay/total/Val", "Delay/total/Val", "Mean_delay/fix/Val", "Std_delay/fix/Val", "Delay/fix/Val", "Mean_delay/blink/Val", "Std_delay/blink/Val", "Delay/blink/Val"]:
                    metrics_class.pop(key)
            add_metrics_to_writer(metrics_class, epoch, writer)


        validation_losses_regress.append(metrics["Loss/regress/Val"])
        validation_losses_class.append(metrics["Loss/class/Val"])
        
        if scheduler:
            scheduler.step(metrics["Loss/regress/Val"])

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.linear_classes.state_dict(), str(dir_checkpoint+"last_model_class.pth"))
            torch.save(model.linear_duration.state_dict(), str(dir_checkpoint+"last_model_regress.pth"))
            if min(validation_losses_regress)==validation_losses_regress[-1]:
                torch.save(model.linear_duration.state_dict(), str(dir_checkpoint+"best_model_regress.pth"))
            if min(validation_losses_class)==validation_losses_class[-1]:
                torch.save(model.linear_classes.state_dict(), str(dir_checkpoint+"best_model_class.pth"))
        
        if early_stop:
            if epoch>early_stop:
                if min(validation_losses_regress[-early_stop:]) >= min(validation_losses_regress[:-early_stop]):
                    early_stop_regress = True
                if min(validation_losses_class[-early_stop:]) >= min(validation_losses_class[:-early_stop]):
                    early_stop_class = True
                if early_stop_class and early_stop_regress:
                    print(f"Early stop on epoch {epoch} (of {epochs})")
                    break

def scheduler_to_str(scheduler):
    if scheduler:
        data_dict = scheduler.__dict__
        return str({"name": str(scheduler), "patience": data_dict["patience"], "mode": data_dict["mode"]})
    else:
        return scheduler