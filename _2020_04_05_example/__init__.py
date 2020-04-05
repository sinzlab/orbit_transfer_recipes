transfer_experiments_, experiments_ = {}, {}

from .baseline import transfer_experiments, experiments
transfer_experiments_.update(transfer_experiments)
experiments_.update(experiments)
# ... import and add any further recipes you have here

transfer_experiments = transfer_experiments_
experiments = experiments_
