transfer_experiments_, experiments_ = {}, {}

from .baselines import transfer_experiments
transfer_experiments_.update(transfer_experiments)

transfer_experiments = transfer_experiments_
experiments = experiments_
