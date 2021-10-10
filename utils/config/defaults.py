from yacs.config import CfgNode as ConfigNode

config = ConfigNode()

config.device = 'cuda'

config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

config.dataset = ConfigNode()
config.dataset.name = 'CIFAR10'
config.dataset.image_size=224
config.dataset.n_classes = 10
config.dataset.dataset_dir = ''

config.model = ConfigNode()
config.model.name = 'resnet_preact'
config.model.type="imagenet"
config.model.backbone = 'resnet_preact'
config.model.init_mode = 'pretrain'
config.model.features_dim=128

config.model.resnet = ConfigNode()
config.model.resnet.depth = 110 
config.model.resnet_preact = ConfigNode()
config.model.resnet_preact.n_blocks = [2, 2, 2, 2]  # for imagenet type model
config.model.resnet_preact.block_type = 'basic'

config.model.efficientnet = ConfigNode()
config.model.efficientnet.subname = 'b0'

config.model.resnext = ConfigNode()
config.model.resnext.depth = 29  # for cifar type model
config.model.resnext.n_blocks = [3, 4, 6, 3]  # for imagenet type model
config.model.resnext.initial_channels = 64
config.model.resnext.cardinality = 8
config.model.resnext.base_channels = 4

config.train = ConfigNode()
config.train.checkpoint = ''
config.train.resume = False
config.train.use_apex = True
config.train.precision = 'O0'
# optimization level for NVIDIA apex
# O0 = fp32
# O1 = mixed precision
# O2 = almost fp16
# O3 = fp16
config.train.batch_size = 128
config.train.optimizer = 'sgd'
config.train.base_lr = 0.1
config.train.momentum = 0.9
config.train.nesterov = True
config.train.weight_decay = 1e-4
config.train.no_weight_decay_on_bn = False
config.train.gradient_clip = 0.0
config.train.start_epoch = 0
config.train.seed = 0
config.train.val_first = True
config.train.val_period = 1
config.train.val_ratio = 0.0
config.train.use_test_as_val = True

config.train.output_dir = 'experiments/exp00'
config.train.log_period = 100
config.train.checkpoint_period = 10

#distribute
config.train.distributed = False

config.train.use_tensorboard = True
config.tensorboard = ConfigNode()
config.tensorboard.train_images = False
config.tensorboard.val_images = False
config.tensorboard.model_params = False

# optimizer
config.optim = ConfigNode()
# Adam
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)
# LARS
config.optim.lars = ConfigNode()
config.optim.lars.eps = 1e-9
config.optim.lars.threshold = 1e-2
# AdaBound
config.optim.adabound = ConfigNode()
config.optim.adabound.betas = (0.9, 0.999)
config.optim.adabound.final_lr = 0.1
config.optim.adabound.gamma = 1e-3

# scheduler
config.scheduler = ConfigNode()
config.scheduler.epochs = 160
# warm up (options: none, linear, exponential)
config.scheduler.warmup = ConfigNode()
config.scheduler.warmup.type = 'none'
config.scheduler.warmup.epochs = 0
config.scheduler.warmup.start_factor = 1e-3
config.scheduler.warmup.exponent = 4
# main scheduler (options: constant, linear, multistep, cosine, sgdr)
config.scheduler.type = 'multistep'
config.scheduler.milestones = [80, 120]
config.scheduler.lr_decay = 0.1
config.scheduler.lr_min_factor = 0.001
config.scheduler.T0 = 10

# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = False
config.train.dataloader.non_blocking = False

# validation data loader
config.validation = ConfigNode()
config.validation.batch_size = 256
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.drop_last = False
config.validation.dataloader.pin_memory = False
config.validation.dataloader.non_blocking = False

config.augmentation = ConfigNode()
config.augmentation.use_albumentations = False

config.test = ConfigNode()
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 256
# test data loader
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False

config.criterion_kwargs=ConfigNode()
config.criterion_kwargs.temperature=0.9
config.criterion_kwargs.num_neighbors=200


def get_default_config():
    return config.clone()