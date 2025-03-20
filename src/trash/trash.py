import torch



path_to_swap_src = "/home/scala/projects/GNN_ContinualLerning/data/openalex/snapshot_0/heterodata/edgelists/author_writes_paper.pt"
path_to_swap_dst = "/home/scala/projects/GNN_ContinualLerning/data/openalex/snapshot_0/heterodata/edgelists/paper_is_written_by_author.pt"


edge_index = torch.load(path_to_swap_src)

reversed_edge_index = torch.zeros(edge_index.shape, dtype=edge_index.dtype)
reversed_edge_index[0] = edge_index[1]
reversed_edge_index[1] = edge_index[0]

reversed_edge_index = torch.nan_to_num(reversed_edge_index)

torch.save(reversed_edge_index, path_to_swap_dst)


