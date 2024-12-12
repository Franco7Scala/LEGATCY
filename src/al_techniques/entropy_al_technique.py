import torch
import numpy

from src.al_techniques.abstract_al_technique import AbstractALTechnique


class EntropyALTechnique(AbstractALTechnique):

    def get_score(self, sample):
        preds = torch.nn.functional.softmax(self.model(input), dim=1).cpu().numpy()
        return (numpy.log(preds + 1e-6) * preds).sum(axis=1)
