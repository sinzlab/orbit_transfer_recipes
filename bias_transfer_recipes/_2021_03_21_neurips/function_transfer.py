transfer_experiments_, experiments_ = {}, {}

from .fd_cov import transfer_experiments
transfer_experiments_.update(transfer_experiments)

from .kd import transfer_experiments
transfer_experiments_.update(transfer_experiments)

from .finetune import transfer_experiments
transfer_experiments_.update(transfer_experiments)

transfer_experiments = transfer_experiments_
experiments = experiments_
