transfer_experiments_, experiments_ = {}, {}

from .direct_training import transfer_experiments
transfer_experiments_.update(transfer_experiments)

from .baselines import transfer_experiments
transfer_experiments_.update(transfer_experiments)

from .frcl import transfer_experiments
transfer_experiments_.update(transfer_experiments)
#
from .fromp import transfer_experiments
transfer_experiments_.update(transfer_experiments)

transfer_experiments = transfer_experiments_
experiments = experiments_
