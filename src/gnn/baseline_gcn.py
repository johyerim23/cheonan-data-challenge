from pathlib import Path

import pandas as pd

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ============================================================
# 경로
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

GNN_DIR = ROOT_DIR / "data" / "gnn"

NODE_PATH = GNN_DIR / "gnn_nodes.csv"
EDGE_PATH = GNN_DIR / "gnn_edges.csv"


# ============================================================
# Graph 불러오기
# ============================================================

nodes = pd.read_csv(
    NODE_PATH
)

edges = pd.read_csv(
    EDGE_PATH
)


source = edges["source"].tolist()
target = edges["target"].tolist()


edge_index = torch.tensor(
    [
        source + target,
        target + source,
    ],
    dtype=torch.long,
)


# ============================================================
# 임시 Node Feature
# ============================================================

# 아직 통합 Feature가 완성되지 않았으므로
# GCN 구조 테스트용 random feature 사용
#
# Node = 31
# Feature = 5개

NUM_NODES = len(nodes)
NUM_FEATURES = 5


torch.manual_seed(42)


x = torch.rand(
    (
        NUM_NODES,
        NUM_FEATURES,
    ),
    dtype=torch.float,
)


# ============================================================
# PyG Data
# ============================================================

data = Data(
    x=x,
    edge_index=edge_index,
)


print()
print("==============================")
print("Graph Data")
print("==============================")

print(data)

print(
    "x shape:",
    data.x.shape
)

print(
    "edge_index shape:",
    data.edge_index.shape
)


# ============================================================
# GCN 모델
# ============================================================

class EmergencyGCN(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=16,
    ):

        super().__init__()


        self.conv1 = GCNConv(
            input_dim,
            hidden_dim,
        )


        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
        )


        self.output = torch.nn.Linear(
            hidden_dim,
            1,
        )


    def forward(
        self,
        x,
        edge_index,
    ):

        # 첫 번째 GCN
        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)


        # 두 번째 GCN
        x = self.conv2(
            x,
            edge_index,
        )

        x = F.relu(x)


        # 각 Node마다 값 하나 출력
        x = self.output(x)


        # 응급수요가 음수가 되지 않도록
        x = F.softplus(x)


        return x.squeeze(-1)


# ============================================================
# 모델 생성
# ============================================================

model = EmergencyGCN(
    input_dim=NUM_FEATURES,
    hidden_dim=16,
)


# ============================================================
# Forward Test
# ============================================================

model.eval()


with torch.no_grad():

    pred = model(
        data.x,
        data.edge_index,
    )


print()
print("==============================")
print("GCN Forward Test")
print("==============================")

print(
    "예측 shape:",
    pred.shape
)


print()
print("행정동별 임시 출력")


for dong, value in zip(
    nodes["행정동"],
    pred,
):

    print(
        f"{dong:8s}: "
        f"{value.item():.4f}"
    )


# ============================================================
# 검증
# ============================================================

assert pred.shape == torch.Size(
    [31]
)


print()
print("GCN Forward Test 성공")