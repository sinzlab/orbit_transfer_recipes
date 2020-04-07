transfer_experiments_, experiments_ = {}, {}
from .resnet_vs_vgg import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)
from .tiny_imagenet import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)
from .representation_analysis import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)

transfer_experiments = transfer_experiments_
experiments = experiments_
