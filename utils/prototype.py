from dataloader.dataloader import create_dataloader
from MemoryBank import MemoryBank
from models import build_model
import torchvision
import torch.nn.functional as F

from scipy import spatial


def fill_memory_bank(loader,model,memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))

#refer to google research
# distribution alginment  
class PMovingAverage:
    def __init__(self, name, nclass, buf_size):
        # MEAN aggregation is used by DistributionStrategy to aggregate
        # variable updates across shards
        self.ma = tf.Variable(tf.ones([buf_size, nclass]) / nclass,
                              trainable=False,
                              name=name,
                              aggregation=tf.VariableAggregation.MEAN)

    def __call__(self):
        v = tf.reduce_mean(self.ma, axis=0)
        return v / tf.reduce_sum(v)

    def update(self, entry):
        entry = tf.reduce_mean(entry, axis=0)
        return tf.assign(self.ma, tf.concat([self.ma[1:], [entry]], axis=0))


class PData:
    def __init__(self, dataset: torchvision.datasets):
        self.has_update = False
        if dataset is not None:
            self.p_data = tf.constant(dataset.p_unlabeled, name='p_data')
        else:
            # MEAN aggregation is used by DistributionStrategy to aggregate
            # variable updates across shards
            self.p_data = renorm(tf.ones([dataset.nclass]))
            self.has_update = True

    def __call__(self):
        return self.p_data / tf.reduce_sum(self.p_data)

    def update(self, entry, decay=0.999):
        entry = tf.reduce_mean(entry, axis=0)
        return tf.assign(self.p_data, self.p_data * decay + entry * (1 - decay))
#reproduce Semi-supervised Contrastive Learning with Similarity Co-calibration
#similarity distribution
def predefined_prototype(config,model):
    labalbed_dataloader = create_dataloader(config,False,True)
    memory_bank_base = MemoryBank(len(labalbed_dataloader), 
                                config['model']['features_dim'],
                                config['dataset']['n_classes'], config['criterion_kwargs']['temperature'])

    if config.device=="cuda":
        memory_bank_base.cuda()
    # Fill memory bank
    print('Fill memory bank for kNN...')
    fill_memory_bank(labalbed_dataloader, model, memory_bank_base)
    #calcuate the prefined_prototype 
    return memory_bank_base

def label_assignment(config,p_data,p_model):
    unlabeled_dataloader=create_dataloader(config,False,False)
    model = build_model(config)
    memory_bank_base = predefined_prototype(config,model) 
    memory_bank_unlabeled = MemoryBank(len(unlabeled_dataloader), 
                                config['model']['features_dim'],
                                config['dataset']['n_classes'], config['criterion_kwargs']['temperature'])
    logits_y=[]
    softmax_y=[]
    for batch in unlabeled_dataloader:
        feature_vector=model(batch)
        logits_y.extend(feature_vector)
        softmax_y.extend((F.softmax(feature_vector,dim=1)))
    p_ratio = (1e-6 + p_data) / (1e-6 + p_model)
    p_weighted = p_model_y * p_ratio
    p_weighted /= tf.reduce_sum(p_weighted, axis=1, keep_dims=True)

    p_model=PMovingAverage(softmax_y)
    #calculate cosine similarity
    feature_vector =feature_vector.expand(memory_bank_base.features.size(1))
    #label guess
    p_s = 1 - spatial.distance.cosine(feature_vector,)
    p_s_h = p_s.mean()
    memory_bank_unlabeled.update(p_s_h,None)



def prototype_mixture(config):
    #build labeled dataset distribution
    #change it to dataset?
    labalbed_dataloader = create_dataloader(config,False,True)
    p_data=PData(labalbed_dataloader)
    #build unlabeled distribution
    unlabeled_dataloader=create_dataloader(config,False,False)
    p_model=PMovingAverage(unlabeled_dataloader)
    return




    

if __name__ == '__main__':

    from utils.common_utils import create_config
    import argparse
    # Parser
    parser = argparse.ArgumentParser(description='ssl_base_model')
    parser.add_argument('--config_env',
                        help='Config file for the environment')
    args = parser.parse_args()
    p=create_config(args.config_exp)
    predefined_prototype(config)