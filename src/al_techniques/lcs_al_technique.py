from src.al_techniques.abstract_al_technique import AbstractALTechnique


class LCSALTechnique(AbstractALTechnique):

    def get_score(self, samples, target_type):
        return self.model(samples.x_dict, samples.edge_index_dict)[target_type].max(axis=1).values.cpu().detach().numpy()
