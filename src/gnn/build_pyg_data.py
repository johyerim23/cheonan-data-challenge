from pathlib import Path

import pandas as pd
import torch

from torch_geometric.data import Data


# ============================================================
# 경로
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

GNN_DIR = ROOT_DIR / "data" / "gnn"

NODE_PATH = GNN_DIR / "gnn_nodes.csv"
EDGE_PATH = GNN_DIR / "gnn_edges.csv"


# ============================================================
# CSV 로드
# ============================================================

nodes = pd.read_csv(
    NODE_PATH
)

edges = pd.read_csv(
    EDGE_PATH
)


print("Node 수:", len(nodes))
print("무방향 Edge 수:", len(edges))


# ============================================================
# PyG edge_index 생성
# ============================================================

source = edges["source"].tolist()
target = edges["target"].tolist()


# 현재 CSV에는
#
# A - B
#
# 가 한 번만 들어있음.
#
# GCN에서는 무방향 그래프로 사용하므로
#
# A -> B
# B -> A
#
# 두 방향을 모두 생성

edge_index = torch.tensor(
    [
        source + target,
        target + source,
    ],
    dtype=torch.long,
)


# ============================================================
# PyG Data 객체
# ============================================================

data = Data(
    edge_index=edge_index,
    num_nodes=len(nodes),
)


# ============================================================
# 결과 확인
# ============================================================

print()
print("==============================")
print("PyTorch Geometric Graph")
print("==============================")

print(data)

print(
    "edge_index shape:",
    data.edge_index.shape
)

print(
    "num_nodes:",
    data.num_nodes
)


# 간단 검증
assert data.num_nodes == 31

assert data.edge_index.shape[0] == 2

assert (
    data.edge_index.shape[1]
    == 2 * len(edges)
)


print()
print("PyG 그래프 변환 성공")