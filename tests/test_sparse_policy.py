import torch

from attention_backends.sparse import MatrixSparseBackend, SparseConfig


def test_sparse_bootstrap_adds_first_frame_key_sink(monkeypatch):
    import attention_backends.sparse as sparse_module

    monkeypatch.setattr(
        sparse_module,
        "routing_context",
        lambda include_grid=False: (1, 0, 0, (2, 2, 2))
        if include_grid
        else (1, 0, 0),
    )

    def dense_route(q, k, v, layout, query_chunk, target, keep, scale):
        del k, layout, query_chunk, target, keep, scale
        return {
            "mask": torch.zeros(1, 4, 4, dtype=torch.bool),
            "mass": torch.zeros(1, 4, 4),
        }, torch.zeros_like(v)

    monkeypatch.setattr(sparse_module, "_dense_route", dense_route)
    backend = MatrixSparseBackend(SparseConfig(tile=2))
    tensor = torch.randn(1, 8, 1, 4)
    backend(tensor, tensor, tensor)

    route = next(iter(backend.state.values()))["mask"]
    assert route[..., :2].all()
    assert not route[..., 2:].any()


def test_sparse_first_layer_is_always_dense(monkeypatch):
    import attention_backends.sparse as sparse_module

    monkeypatch.setattr(
        sparse_module,
        "routing_context",
        lambda include_grid=False: (0, 2, 1, (2, 2, 2))
        if include_grid
        else (0, 2, 1),
    )
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    backend = MatrixSparseBackend()
    output = backend(q, k, v, dtype=torch.float32)
    reference = backend.dense(q, k, v, dtype=torch.float32)
    torch.testing.assert_close(output, reference)
    assert backend.state == {}
