import random

from src.al_techniques.abstract_al_technique import AbstractALTechnique


class RandomALTechnique(AbstractALTechnique):

    def get_score(self, sample):
        return random.uniform(0, 1)
