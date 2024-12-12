from src.al_techniques.abstract_al_technique import AbstractALTechnique


class LCSALTechnique(AbstractALTechnique):

    def get_score(self, sample):
        return self.model(sample).max(axis=1).values.cpu().numpy()
