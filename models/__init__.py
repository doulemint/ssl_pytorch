
def get_encoder(backbone):
    model=None
    return model

def build_model(config):
    backbone= get_encoder(config['model']['backbond'])
    if config['model']['name']=='simclr':
        from models import ContrastiveModel
        model = ContrastiveModel(backbone,)
    return model