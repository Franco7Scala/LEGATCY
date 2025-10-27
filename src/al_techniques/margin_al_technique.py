import torch

from src.al_techniques.abstract_al_technique import AbstractALTechnique
from src.support.utils import get_time_in_millis


class MarginALTechnique(AbstractALTechnique):

    def get_score(self, sample, target_type):
        #st = get_time_in_millis()
        fw = self.model(sample.x_dict, sample.edge_index_dict)[0][target_type]
        #print(f"time forward: {get_time_in_millis() - st}")
        #st = get_time_in_millis()
        preds = torch.nn.functional.softmax(fw, dim=1)
        #print(f"time softmax: {get_time_in_millis() - st}")
        #st = get_time_in_millis()
        preds_argmax = torch.argmax(preds, dim=1)
        #print(f"time argmax: {get_time_in_millis() - st}")
        #st = get_time_in_millis()
        max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
        #print(f"time preds: {get_time_in_millis() - st}")
        #st = get_time_in_millis()
        preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
        #print(f"time ones: {get_time_in_millis() - st}")
        #st = get_time_in_millis()
        preds_sub_argmax = torch.argmax(preds, dim=1)
        #print(f"time argmax: {get_time_in_millis() - st}")
        #st = get_time_in_millis()
        aaa = (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().detach().numpy()
        #print(f"time max_preds: {get_time_in_millis() - st}")
        #st = get_time_in_millis()
        return aaa
