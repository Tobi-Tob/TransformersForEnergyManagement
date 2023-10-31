from rewards.comfort_reward import ComfortRewardFunction
from rewards.mixed_reward import MixReward
from rewards.custom_reward import TempDiffReward, UnservedEnergyReward, CombinedReward
from rewards.weighted_reward import WeightedRewardFunction

###################################################################
#####                Specify your reward here                 #####
###################################################################

SubmissionReward = CombinedReward
