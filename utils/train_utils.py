





import torch
from dataloader.dataloader import *
from .prototype import *


def train(config,model,train_loader,optimizer,criterion,scheduler,logger,epoch):
    device=config.device
    model.to(device)
    model.train()

    logger.info(f"running Train {epoch}")
    # print(f"running Train {epoch}")
    
    running_accuracy=0.0
    for step, batch in enumerate(train_loader):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        #clean gradients parameter
        optimizer.zero_grad()

        _,logits_y=model(images)
        loss = criterion(logits_y, targets)
        loss.backward()

        optimizer.step()

        pred = torch.max(logits_y,dim=1)[1] 
        #todo: handle one hot label

        running_accuracy +=(pred==targets).sum().item()
        if step%200==0:
            logger.info(f"Accy-top1: {running_accuracy}")

    logger.info(f"Accy-top1: {running_accuracy}")

    scheduler.step()
    return loss

#p_model: runing similairty distribution
#emb_sums: predefined protetype 
def unlabeled_train(config,p_model,criterion,unlabeled_dataloader,model,num_neighbors,emb_sums,logger):
    num_neighbors = 20
    device = config.device
    memory_bank_unlabeled = MemoryBank(len(unlabeled_dataloader)*config.train.batch_size, 
                                config.dataset.n_classes,
                                config.dataset.n_classes, config.criterion_kwargs.temperature)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)


    for batch,filenames in unlabeled_dataloader:
        batch = batch.to(device)
        feature_vector,logits_y=model(batch)
        p_s=torch.zeros(feature_vector.size(0), config.dataset.n_classes)
        #if batch_size>0 we have to calculate them one by one
        if len(feature_vector.size())>=2:
            for i,fv in enumerate(feature_vector):
                fv = torch.unsqueeze(fv,dim=0)
        #calculate cosine similarity
                p_s[i] = cos(fv,emb_sums)
        else:
            p_s[0] = cos(feature_vector,emb_sums)
        memory_bank_unlabeled.update(p_s,torch.tensor([0]*feature_vector.size(0)),filenames)
        #distribution alignment
        targets_u = logits_y.detach().cpu()* p_model
        #sharpening
        targets_u = targets_u**(1/0.5)
        targets_u = targets_u / targets_u.sum(dim=1, keepdim=True)
        pseudo_label = F.normalize(targets_u.detach(), dim = 1)
        #loss
        losses,mask = criterion(logits_y, pseudo_label)
        print("losses: ",losses)
        p_model.update(p_s)
    
    return losses,memory_bank_unlabeled,mask,p_model

#in contrastive train model_q only is maintained by positive selection
def contrastive_train(config,model,train_loader,optimizer,criterion,scheduler,epoch,negative_queue,emb_sum):
    device=config.device
    model.to(device)
    model.train()
    #todo:logger
    print(f"running Train {epoch}")
    
    running_accuracy=0.0
    for step, batch in enumerate(train_loader):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        #clean gradients parameter
        optimizer.zero_grad()

        feature_vector,logits_y=model(images)
        #
        sim = (feature_vector,emb_sum)
        loss = criterion(logits_y, targets)
        # loss.backward()

        optimizer.step()

        pred = torch.max(logits_y,dim=1)[1] 
        #todo: handle one hot label

        running_accuracy +=(pred==targets).sum().item()
    
    
    
    #train and calculate unlabeled_dataset

    
    scheduler.step()
    return loss
def mine_nn(memory_bank_unlabeled,num_neighbors):
    _, pred = memory_bank_unlabeled.features.topk(num_neighbors, 0, False, True)
    pred = pred.t()
    topk_files=[]
    topk_labels=[]
    for labels,index in enumerate(pred):
        topk_files.extend([memory_bank_unlabeled.filenames[i] for i in index])
        topk_labels.extend([labels]*num_neighbors)
    unlabeled_df = pd.DataFrame({"filename": topk_files, "label": topk_labels})
    return unlabeled_df

def get_positive_sample(memory_bank_unlabeled,num_neighbors,mask):
    unlabeled_df = mine_nn(memory_bank_unlabeled,num_neighbors)
    constrastive_dataloader = create_contrastive_dataloader(config,True,unlabeled_df)
    return constrastive_dataloader

#we didn't consider pr_loss result.. simply mined samples
def get_mixup(memory_bank_unlabeled,num_neighbors):
    #mine the nearest nearbor
    #convert topk to df
    # pred size(num_neighbors x n_classes)
    unlabeled_df = mine_nn(memory_bank_unlabeled,num_neighbors)
    
    # prototype_mixture update new embeddings
    Mixup_dataloader = create_mixup_dataloader(config,True,unlabeled_df)
    # emb_sums = predefined_prototype(config,model,Mixup_dataloader)
    return Mixup_dataloader 

def val(config,model,train_loader,criterion,epoch):
    device=config.device
    model.to(device)
    model.val()
    #todo:logger
    print(f"running Train {epoch}")
    
    running_accuracy=0.0
    for step, batch in enumerate(train_loader):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        feature_vector,logits_y=model(images)
        loss = criterion(logits_y, targets)

        pred = torch.max(logits_y,dim=1)[1] 
        #todo: handle one hot label

        running_accuracy +=(pred==targets).sum().item()

