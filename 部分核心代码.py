"""
核心代码：基于多源遥感特征的桉树冠层高度反演

说明：
本代码用于展示本文模型的主要实现过程，包括：
1. 多源遥感中心像元特征构建
2. 残差增强MLP模型
3. Huber损失训练
4. 验证集精度评价

为保护项目数据及完整工程实现，数据读取、数据预处理、
样本构建及全区域批量反演等部分未在此代码中展示。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# 1. 基本参数
# ============================================================

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT = 0.3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_TREE_HEIGHT = 40.0


# ============================================================
# 2. 多源遥感特征
# ============================================================

FEATURE_NAMES = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
    'B11', 'B12',
    'VV', 'VH',
    'Elevation',
    'NDVI', 'EVI', 'NDRE',
    'GNDVI', 'SAVI',
    'Slope', 'Aspect'
]


# ============================================================
# 3. 植被指数计算
# ============================================================

def calculate_vegetation_indices(pixel):
    """
    根据中心像元的Sentinel-2光谱反射率计算植被指数。
    """

    eps = 1e-6

    blue = pixel[0]
    green = pixel[1]
    red = pixel[2]
    nir = pixel[6]
    swir1 = pixel[7]

    ndvi = (nir - red) / (nir + red + eps)

    evi = (
        2.5 * (nir - red) /
        (nir + 6 * red - 7.5 * blue + 1 + eps)
    )

    ndre = (
        nir - swir1
    ) / (
        nir + swir1 + eps
    )

    gndvi = (
        nir - green
    ) / (
        nir + green + eps
    )

    savi = (
        1.5 * (nir - red)
    ) / (
        nir + red + 0.5 + eps
    )

    return np.array([
        ndvi,
        evi,
        ndre,
        gndvi,
        savi
    ], dtype=np.float32)


# ============================================================
# 4. 地形特征计算
# ============================================================

def calculate_terrain_features(elevation_patch):
    """
    根据DEM邻域计算中心像元的坡度和坡向。
    """

    dx, dy = np.gradient(elevation_patch)

    center_y = elevation_patch.shape[0] // 2
    center_x = elevation_patch.shape[1] // 2

    slope = np.arctan(
        np.sqrt(
            dx[center_y, center_x] ** 2 +
            dy[center_y, center_x] ** 2
        )
    )

    aspect = np.arctan2(
        -dx[center_y, center_x],
        dy[center_y, center_x]
    )

    slope = slope / (np.pi / 2)

    aspect = (
        aspect + np.pi
    ) / (2 * np.pi)

    return np.array(
        [slope, aspect],
        dtype=np.float32
    )


# ============================================================
# 5. 中心像元特征构建
# ============================================================

def build_pixel_features(pixel,
                         elevation_patch):
    """
    构建GEDI足迹中心对应像元的多源遥感特征。

    实际数据读取和空间配准过程省略。
    """

    vegetation_features = \
        calculate_vegetation_indices(pixel)

    terrain_features = \
        calculate_terrain_features(
            elevation_patch
        )

    features = np.concatenate([
        pixel,
        vegetation_features,
        terrain_features
    ])

    return features.astype(np.float32)


# ============================================================
# 6. 数据集
# ============================================================

class TreeHeightDataset(Dataset):

    def __init__(self,
                 features,
                 heights):

        self.features = features
        self.heights = heights

        self.scaler = StandardScaler()

        self.features = \
            self.scaler.fit_transform(
                self.features
            )

    def __len__(self):
        return len(self.heights)

    def __getitem__(self, index):

        x = torch.tensor(
            self.features[index],
            dtype=torch.float32
        )

        y = torch.tensor(
            self.heights[index],
            dtype=torch.float32
        )

        return x, y


# ============================================================
# 7. 残差MLP模块
# ============================================================

class ResidualMLPBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout=0.3):

        super().__init__()

        self.fc1 = nn.Linear(
            in_dim,
            out_dim
        )

        self.bn1 = nn.BatchNorm1d(
            out_dim
        )

        self.fc2 = nn.Linear(
            out_dim,
            out_dim
        )

        self.bn2 = nn.BatchNorm1d(
            out_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        if in_dim != out_dim:
            self.shortcut = nn.Linear(
                in_dim,
                out_dim
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        residual = self.shortcut(x)

        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out = out + residual

        return F.relu(out)


# ============================================================
# 8. 残差增强MLP
# ============================================================

class EnhancedTreeHeightMLP(nn.Module):

    def __init__(self,
                 input_dim,
                 dropout=0.3):

        super().__init__()

        self.input_layer = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(dropout)
        )

        self.block1 = ResidualMLPBlock(
            256,
            512,
            dropout
        )

        self.block2 = ResidualMLPBlock(
            512,
            256,
            dropout
        )

        self.block3 = ResidualMLPBlock(
            256,
            128,
            dropout
        )

        self.output_layer = nn.Linear(
            128,
            1
        )

    def forward(self, x):

        x = self.input_layer(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        return self.output_layer(x).squeeze(-1)


# ============================================================
# 9. 模型训练
# ============================================================

def train_model(train_loader,
                val_loader,
                input_dim):

    model = EnhancedTreeHeightMLP(
        input_dim=input_dim,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.HuberLoss(
        delta=1.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    best_rmse = np.inf

    for epoch in range(EPOCHS):

        # -------------------------
        # Training
        # -------------------------

        model.train()

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(
                pred,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        # -------------------------
        # Validation
        # -------------------------

        model.eval()

        predictions = []
        observations = []

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(DEVICE)

                pred = model(x)

                predictions.extend(
                    pred.cpu().numpy()
                )

                observations.extend(
                    y.numpy()
                )

        predictions = np.asarray(
            predictions
        )

        observations = np.asarray(
            observations
        )

        rmse = np.sqrt(
            mean_squared_error(
                observations,
                predictions
            )
        )

        mae = mean_absolute_error(
            observations,
            predictions
        )

        r2 = r2_score(
            observations,
            predictions
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"RMSE={rmse:.3f} m | "
            f"MAE={mae:.3f} m | "
            f"R²={r2:.3f}"
        )

        # -------------------------
        # 保存最优模型
        # -------------------------

        if rmse < best_rmse:

            best_rmse = rmse

            torch.save(
                model.state_dict(),
                "best_tree_height_mlp.pth"
            )

    return model


# ============================================================
# 10. 模型评价
# ============================================================

def evaluate_model(model,
                   data_loader):

    model.eval()

    predictions = []
    observations = []

    with torch.no_grad():

        for x, y in data_loader:

            x = x.to(DEVICE)

            pred = model(x)

            predictions.extend(
                pred.cpu().numpy()
            )

            observations.extend(
                y.numpy()
            )

    predictions = np.asarray(
        predictions
    )

    observations = np.asarray(
        observations
    )

    rmse = np.sqrt(
        mean_squared_error(
            observations,
            predictions
        )
    )

    mae = mean_absolute_error(
        observations,
        predictions
    )

    r2 = r2_score(
        observations,
        predictions
    )

    bias = np.mean(
        predictions - observations
    )

    print("\n==============================")
    print("模型验证结果")
    print("==============================")
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.4f} m")
    print(f"MAE  : {mae:.4f} m")
    print(f"Bias : {bias:.4f} m")

    return r2, rmse, mae, bias