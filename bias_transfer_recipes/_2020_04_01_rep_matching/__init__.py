transfer_experiments_, experiments_ = {}, {}
from .baseline import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)
from .noise_adv_training import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)
from .representation_analysis import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)
from .representation_matching import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)
from .representation_matching_noise_adv import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)
from .self_attention import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)
transfer_experiments = transfer_experiments_
experiments = experiments_
