import torch

from src.al_techniques.abstract_al_technique import AbstractALTechnique


class MarginALTechnique(AbstractALTechnique):

    def get_score(self, sample):
        preds = torch.nn.functional.softmax(self.model(sample), dim=1)
        preds_argmax = torch.argmax(preds, dim=1)
        max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
        preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
        preds_sub_argmax = torch.argmax(preds, dim=1)
        return (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy()
