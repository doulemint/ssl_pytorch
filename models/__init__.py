
from logging import raiseExceptions
import timm
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_encoder(configs):
    pretrain=False
    if configs.model.init_mode=='pretrain':
        pretrain=True
    model=None
    if configs.model.name.startswith("resnext"):
        # model = torchvision.models.resnext50_32x4d(pretrained=True)
        # model.avgpool = nn.AdaptiveAvgPool2d(1)
        # model.fc = nn.Linear(2048, configs.num_classes)
        ####
        model = timm.create_model('resnext50_32x4d', pretrained=pretrain)
        # set_parameter_requires_grad(model, feature_extract)
        n_features = model.fc.in_features
        model.fc = Identity() #remove the lastest fc layer
        # model.fc = nn.Linear(n_features, configs.dataset.n_classes)
    elif configs.model.name.startswith("resnet50"):
        model = timm.create_model("resnet50", pretrained=pretrain)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        n_features = model.fc.in_features
        model.fc = Identity()
        # model.fc = nn.Linear(model.fc.in_features, configs.dataset.n_classes)
    elif configs.model.name.startswith("efficientnet-b5"):
        model = timm.create_model('tf_efficientnet_b5_ns',drop_rate=0.7, pretrained=pretrain, drop_path_rate=0.2)
        # set_parameter_requires_grad(model, feature_extract)
        n_features = model.classifier.in_features
        model.classifier = Identity()
        # model.classifier = nn.Linear(model.classifier.in_features, configs.dataset.n_classes)

    else:
        ValueError("unknow model name {}".format(configs.model.name))
    return {"backbone":model,"dim":n_features}

def build_model(config,type="query",head="mlp"):
    backbone= get_encoder(config)
    if type=='contra':
        from .models import ContrastiveModel
        model = ContrastiveModel(backbone,head=head,features_dim=config.model.features_dim)
    else:
        from .models import MultiheadModel
        model = MultiheadModel(backbone,config.dataset.n_classes,head=head,features_dim=config.model.features_dim)
    return model