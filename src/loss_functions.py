import torch
from torch.nn import MSELoss, CrossEntropyLoss

def custom_loss(classes_prediction, duration_prediction, tgt_y, reduct=False):
    MSE = MSELoss(reduction="sum")
    CE = CrossEntropyLoss(reduction="sum")

    duration = tgt_y[:,-1,1].squeeze()
    classes = tgt_y[:,-1,0].squeeze()
    classes_prediction = classes_prediction[:,-1].squeeze()
    duration_prediction = duration_prediction[:,-1].squeeze()

    # duration = tgt_y[:,:,1].squeeze()
    # classes = tgt_y[:,:,0].squeeze().type(torch.LongTensor).to(tgt_y.device)
    # classes_prediction = classes_prediction[:,:].squeeze().transpose(1,2)#.argmax(-1)
    # duration_prediction = duration_prediction[:,:].squeeze()

    mse = MSE(duration_prediction, duration)
    # ce = CE(classes_prediction.type(torch.int64), classes.type(torch.int64))
    ce = CE(classes_prediction, classes.type(torch.LongTensor).to(tgt_y.device))
    if reduct:
        return mse+ce
    else:
        return {"regress":mse, "class":ce}

    # return mse

def custom_loss2(classes_prediction, duration_prediction, tgt_y):
    MSE = MSELoss()
    CE = CrossEntropyLoss()

    duration = tgt_y[:,-1,1].squeeze()
    classes = tgt_y[:,-1,0].squeeze()
    classes_prediction = classes_prediction[:,-1].squeeze()
    duration_prediction = duration_prediction[:,-1].squeeze()

    mse = MSE(duration_prediction, duration)
    ce = CE(classes_prediction, classes.type(torch.int64))
    return ce*mse