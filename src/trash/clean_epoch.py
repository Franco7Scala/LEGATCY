import sys


#path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/FullRetraining/log_1765389711550.log"
#path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/OnlineTraining/log_1765389712553.log"
#path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/VotingStrategy/log_1765389713557.log"
#path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/ActiveERS2/RandomALTechnique/log_1765389707535.log"
#path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/ActiveERS2/MarginALTechnique/log_1765389710547.log"
#path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/ActiveERS2/LCSALTechnique/log_1765389708538.log"
#path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/ActiveERS2/EntropyALTechnique/log_1765389709542.log"

path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/FullRetraining/log_1765792427055.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/OnlineTraining/log_1765792428058.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/VotingStrategy/log_1765792429061.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/ActiveERS2/RandomALTechnique/log_1765792423042.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/ActiveERS2/MarginALTechnique/log_1765792426052.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/ActiveERS2/LCSALTechnique/log_1765792424045.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/ActiveERS2/EntropyALTechnique/log_1765792425048.log"
path = "/Users/francesco/Desktop/log_1765884059427.log"

path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_50_meta_False_rfactor_05/FullRetraining/log_1767123744529.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_50_meta_False_rfactor_05/OnlineTraining/log_1767123745534.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_50_meta_False_rfactor_05/VotingStrategy/log_1767347722895.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_50_meta_False_rfactor_05/ActiveERS2/RandomALTechnique/log_1767347718881.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_50_meta_False_rfactor_05/ActiveERS2/MarginALTechnique/log_1767347721891.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_50_meta_False_rfactor_05/ActiveERS2/LCSALTechnique/log_1767347719884.log"
path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_50_meta_False_rfactor_05/ActiveERS2/EntropyALTechnique/log_1767347720888.log"
#path = "/Users/francesco/Desktop/n_snap_4_n_first_3_k_200_meta_False_rfactor_05/DyHANE/log_1766218772883.log"


with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.startswith("Epoch: ") and not line == "\n" and not line == "":
            print(line, end="")
