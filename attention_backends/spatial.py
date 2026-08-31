"""Spatial token reordering for fixed-size FlexAttention route blocks."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch


def _divisors(value):
    return [candidate for candidate in range(1, value + 1)
            if value % candidate == 0]


def _morton_code(y, x):
    """Interleave integer coordinate bits for a compact spatial traversal."""
    result = 0
    bit = 0
    while (1 << bit) <= max(y, x):
        result |= ((x >> bit) & 1) << (2 * bit)
        result |= ((y >> bit) & 1) << (2 * bit + 1)
        bit += 1
    return result


def _choose_microtile(height, width, block_size, max_tokens):
    candidates = []
    for tile_h in _divisors(height):
        for tile_w in _divisors(width):
            area = tile_h * tile_w
            if area <= max_tokens and block_size % area == 0:
                candidates.append((tile_h, tile_w))
    if not candidates:
        return 1, 1
    # Prefer the largest exact microtile, then the least elongated shape.
    return min(candidates, key=lambda shape: (
        -(shape[0] * shape[1]),
        abs(math.log(shape[1] / shape[0])),
    ))


@dataclass
class SpatialTokenLayout:
    grid: tuple[int, int, int]
    block_size: int
    microtile_shape: tuple[int, int]
    permutation_cpu: torch.Tensor
    inverse_cpu: torch.Tensor
    neighbors_cpu: torch.Tensor
    block_frames_cpu: torch.Tensor

    def __post_init__(self):
        self._device_cache = {}

    @property
    def tokens(self):
        frames, height, width = self.grid
        return frames * height * width

    @property
    def blocks(self):
        return math.ceil(self.tokens / self.block_size)

    @property
    def signature(self):
        return self.grid, self.block_size, self.microtile_shape

    def _device_values(self, device):
        device = torch.device(device)
        if device not in self._device_cache:
            self._device_cache[device] = (
                self.permutation_cpu.to(device),
                self.inverse_cpu.to(device),
                self.neighbors_cpu.to(device),
            )
        return self._device_cache[device]

    def reorder(self, tensor):
        permutation, _, _ = self._device_values(tensor.device)
        return tensor.index_select(1, permutation)

    def restore(self, tensor):
        _, inverse, _ = self._device_values(tensor.device)
        return tensor.index_select(1, inverse)

    def neighbors(self, device):
        return self._device_values(device)[2]


def build_spatial_layout(grid, block_size=128, max_microtile_tokens=32):
    """Build a reversible Morton-ordered packing with no sequence padding.

    Each exact HxW microtile remains contiguous and its area divides the route
    block size, so no microtile is split across route blocks. Route blocks may
    contain several nearby microtiles. A small number can straddle frames when
    a frame is not block-aligned; their neighbour is chosen by majority vote.
    """
    if grid is None or len(grid) != 3:
        return None
    frames, height, width = (int(value) for value in grid)
    if min(frames, height, width, block_size, max_microtile_tokens) <= 0:
        return None
    tile_h, tile_w = _choose_microtile(
        height, width, block_size, max_microtile_tokens)
    area = tile_h * tile_w
    tiles_per_block = block_size // area

    ordered_tiles = []
    tile_order = {}
    tile_rows = height // tile_h
    tile_cols = width // tile_w
    for frame in range(frames):
        frame_tiles = [
            (tile_y, tile_x)
            for tile_y in range(tile_rows)
            for tile_x in range(tile_cols)
        ]
        frame_tiles.sort(key=lambda point: (
            _morton_code(point[0], point[1]), point[0], point[1]))
        for tile_y, tile_x in frame_tiles:
            tile_order[(frame, tile_y, tile_x)] = len(ordered_tiles)
            ordered_tiles.append((frame, tile_y, tile_x))

    permutation = []
    for frame, tile_y, tile_x in ordered_tiles:
        y0, x0 = tile_y * tile_h, tile_x * tile_w
        permutation.extend(
            (frame * height + y) * width + x
            for y in range(y0, y0 + tile_h)
            for x in range(x0, x0 + tile_w)
        )
    permutation_cpu = torch.tensor(permutation, dtype=torch.long)
    inverse_cpu = torch.empty_like(permutation_cpu)
    inverse_cpu[permutation_cpu] = torch.arange(len(permutation_cpu))

    block_tiles = defaultdict(list)
    tile_to_block = {}
    for tile_index, tile in enumerate(ordered_tiles):
        block = tile_index // tiles_per_block
        tile_to_block[tile] = block
        block_tiles[block].append(tile)

    blocks = math.ceil(len(permutation) / block_size)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    neighbors = torch.full((blocks, 4), -1, dtype=torch.int32)
    block_frames = torch.full((blocks,), -1, dtype=torch.int32)
    for block in range(blocks):
        tiles = block_tiles[block]
        frame_counts = Counter(tile[0] for tile in tiles)
        if len(frame_counts) == 1:
            block_frames[block] = next(iter(frame_counts))
        for direction, (dy, dx) in enumerate(directions):
            votes = Counter()
            for frame, tile_y, tile_x in tiles:
                target = (frame, tile_y + dy, tile_x + dx)
                target_block = tile_to_block.get(target)
                if target_block is not None and target_block != block:
                    votes[target_block] += 1
            if votes:
                neighbors[block, direction] = votes.most_common(1)[0][0]

    return SpatialTokenLayout(
        grid=(frames, height, width),
        block_size=block_size,
        microtile_shape=(tile_h, tile_w),
        permutation_cpu=permutation_cpu,
        inverse_cpu=inverse_cpu,
        neighbors_cpu=neighbors,
        block_frames_cpu=block_frames,
    )
