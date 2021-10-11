import torch
from dataloader.dataloader import *
import models

from .prototype import *
from utils.metrics import compute_accuracy


def train(config,model,train_loader,optimizer,criterion,scheduler,logger,epoch):
    device=config.device
    model.to(device)
    model.train()

    logger.info(f"running Train {epoch}")
    # print(f"running Train {epoch}")
    
    running_accuracy=0.0
    running_loss=0.0

    for step, batch in enumerate(train_loader):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        #clean gradients parameter
        optimizer.zero_grad()

        _,logits_y=model(images)
        loss = criterion(logits_y, targets)
        loss.backward()
        running_loss += loss.detach()

        optimizer.step()

        pred = torch.max(logits_y,dim=1)[1] 
        #todo: handle one hot label

        running_accuracy +=(pred==targets).sum().item()
        if step%200==0:
            logger.info(f"{step}/{len(train_loader)} Accy-top1: {running_accuracy/((step+1)*config.train.batch_size)} Loss: {running_accuracy}")

    logger.info(f"{step}/{len(train_loader)} Accy-top1: {running_accuracy/(len(train_loader)*config.train.batch_size)} Loss: {running_accuracy}")

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
    running_loss = 0.0


    for step,(batch,filenames) in enumerate(unlabeled_dataloader):
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
        targets_u = F.softmax(logits_y.detach(),dim=1)
        targets_u = targets_u * p_model().to(device)
        #sharpening
        targets_u = targets_u**(1/0.5)
        targets_u = targets_u / targets_u.sum(dim=1, keepdim=True)
        targets_u = F.normalize(targets_u, dim = 1)
        #loss
        losses,mask = criterion(logits_y, targets_u)
        running_loss += losses[mask].sum().item()
        if step%200==0:
            logger.info(f"{step}/{len(unlabeled_dataloader)} Loss: {running_loss}/{((step+1)*config.train.batch_size)}")

        p_model.update(p_s)
    logger.info(f"{step}/{len(unlabeled_dataloader)} Loss: {running_loss}/{(len(unlabeled_dataloader)*config.train.batch_size)}")

    
    return losses,memory_bank_unlabeled,mask

#in contrastive train model_q only is maintained by positive selection
def contrastive_train(config,model,train_loader,optimizer,criterion,scheduler,epoch,emb_sum,logger):
    device=config.device
    model.to(device)
    model.train()

    logger.info(f"running constrastive Train {epoch}")
    
    running_loss=0.0
    for step, batch in enumerate(train_loader):
        images = batch[0].to(device)
        targets = batch[1].to(device)

        #clean gradients parameter
        optimizer.zero_grad()

        feature_vector=model(images)
        feature_vector = F.normalize(feature_vector, dim = 1)
        #
        # sim = (feature_vector,emb_sum)
        loss = criterion(feature_vector, targets)
        loss.backward()

        optimizer.step()
        running_loss += loss.detach()

        # pred = torch.max(logits_y,dim=1)[1] 
        if step%200==0:
            logger.info(f"{step}/{len(train_loader)} Loss: {running_loss}/{((step+1)*config.train.batch_size)}")

    logger.info(f"{step}/{len(train_loader)} Loss: {running_loss}/{(len(train_loader)*config.train.batch_size)}")
    scheduler.step()

    return loss


#we didn't consider pr_loss result.. simply mined samples

@torch.no_grad()
def val(config,model,val_loader,criterion,epoch,emb_sums,logger):
    device=config.device
    model.to(device)
    model.eval()

    logger.info(f"running val {epoch}")

    memory_bank_val_labeled = MemoryBank(len(val_loader)*config.train.batch_size, 
                                config.model.features_dim,
                                config.dataset.n_classes, config.criterion_kwargs.temperature)
    
    running_accuracy=0.0
    for step, batch in enumerate(val_loader):
            images = batch[0].to(device)
            targets = batch[1].to(device)
            

            feature_vector,logits_y=model(images)
            loss = criterion(logits_y, targets)

            memory_bank_val_labeled.update(feature_vector,targets)

            pred = torch.max(logits_y,dim=1)[1] 
            #todo: handle one hot label

            running_accuracy +=(pred==targets).sum().item()
            running_accuracy +=(pred==targets).sum().item()
            if step%200==0:
                logger.info(f"{step}/{len(val_loader)} Accy-top1: {running_accuracy/((step+1)*config.train.batch_size)} Loss: {running_accuracy}")
            

    logger.info(f"{step}/{len(val_loader)} Accy-top1: {running_accuracy/(len(val_loader)*config.train.batch_size)} Loss: {running_accuracy}")
    acc = memory_bank_val_labeled.get_knn(emb_sums,memory_bank_val_labeled.features)
    logger.info(f"knn accuracy {acc} over {memory_bank_val_labeled.K}")

    return acc


