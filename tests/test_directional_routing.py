import torch

from attention_backends.routing import (
    directional_budget_update,
    frontier_mask,
)


def _line_neighbors(blocks):
    neighbors = torch.full((blocks, 4), -1, dtype=torch.int32)
    for block in range(blocks):
        if block:
            neighbors[block, 2] = block - 1
        if block + 1 < blocks:
            neighbors[block, 3] = block + 1
    return neighbors


def test_frontier_marks_only_selected_cells_with_open_neighbor():
    route = torch.zeros(1, 4, 4, dtype=torch.bool)
    route[:, :, 1:3] = True
    frontier = frontier_mask(route, _line_neighbors(4))
    assert frontier[:, :, 1:3].all()
    assert not frontier[:, :, [0, 3]].any()


def test_directional_update_replaces_low_score_edge_at_fixed_budget():
    blocks = 4
    route = torch.eye(blocks, dtype=torch.bool).unsqueeze(0)
    mass = route.float()
    for row in range(blocks - 1):
        mass[0, row, row + 1] = 2.0
    result = directional_budget_update(
        mass, route, _line_neighbors(blocks), keep=.5,
        persistence=0, expand_q=False, expand_k=True,
        expand_joint=False)
    updated = result["route"]
    assert (updated.sum(-1) == 1).all()
    assert updated[0, 0, 1]
    assert updated[0, 1, 2]
    assert updated[0, 2, 3]
    assert updated[0, 3, 3]
    assert result["added"].sum() == 3
    assert result["dropped"].sum() == 3
