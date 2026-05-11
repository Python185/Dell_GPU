from sklearn import datasets, model_selection, svm, metrics
import numpy as np
import pandas as pd
from typing import Any
import time
from typing import Callable
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm.auto import tqdm, trange
import copy
import matplotlib.pyplot as plt
from matplotlib import patches
from amplify import (
    VariableGenerator,
    Model,
    solve,
    Poly,
    PolyArray,
    equal_to,
    Constraint,
    FixstarsClient,
)
from datetime import timedelta


# Amplify AEのアクセストークンを入力してください
token = "AE/6NtxlCQT8xfn1NJvbtD7QiLOFsvT9tWl"



PIX_SIZE = 28  # 画像1辺のピクセルサイズ

class MnistSvm:
    def __init__(
        self, mask_ratio: float, train_size=1000, test_size=1000, seed=0
    ) -> None:
        """与えられたマスク率に応じてクラスを初期化"""
        self._data, self._label = datasets.fetch_openml(
            "mnist_784", return_X_y=True, parser="auto"
        )
        self._data /= 255  # type: ignore # scaling
        self._model = svm.SVC()
        self._num_features = len(self._data.columns)
        self._num_masked_features = int(self._num_features * mask_ratio)

        self._train_data, self._test_data, self._train_labels, self._test_labels = (
            model_selection.train_test_split(
                self._data,
                self._label,
                test_size=test_size,
                train_size=train_size,
                random_state=seed,
            )
        )
        self._rng = np.random.default_rng(seed)
        np.random.seed(seed)
        self._mask: np.ndarray[bool, Any] = np.array([])

    @property
    def num_masked_features(self) -> int:
        return self._num_masked_features

    def _apply_mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """データに対してマスクを適用（マスクがTrueである要素をdrop）"""
        mask_idx = np.where(self._mask)
        return data.drop(columns=data.columns.to_numpy()[mask_idx].tolist())

    def _fetch_masked_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """学習・テストデータに対してマスクを適用"""
        return (
            self._apply_mask(self._train_data),
            self._apply_mask(self._test_data),
            self._train_labels,
            self._test_labels,
        )

    def _train(self, train_data: pd.DataFrame, train_label: pd.Series):
        """モデルの学習を実行"""
        self._model = svm.SVC()
        self._model.fit(train_data, train_label)

    def _eval(self, test_data: pd.DataFrame, test_label: pd.Series) -> float:
        """学習済みモデルの推論・評価を実行し、正答率を返却"""
        prediction = self._model.predict(test_data)
        return float(metrics.accuracy_score(test_label, prediction))

    def generate_random_mask(self) -> np.ndarray[bool, Any]:
        """マスク率を満たすマスクをランダムに生成"""
        mask = [True] * self.num_masked_features + [False] * (
            self._num_features - self.num_masked_features
        )
        self._rng.shuffle(mask)
        return np.array(mask)

    def train_eval(self, mask: np.ndarray[bool, Any]) -> float:
        """マスク適用後のデータに基づき機械学習を行い、学習後のモデル評価を実施、モデルの正答率を返却"""
        self._mask = mask
        train_data, test_data, train_label, test_label = self._fetch_masked_data()
        self._train(train_data, train_label)
        pred = self._eval(test_data, test_label)
        return pred



# マスク率 0（特徴量をマスクしない）として、MnistSvm をインスタンス化。
mnist_svm = MnistSvm(mask_ratio=0)

accuracy_score = mnist_svm.train_eval(np.array([False] * PIX_SIZE * PIX_SIZE))
print(f"{accuracy_score=:.3f}")



# 問題サイズ（＝最適化対象のマスクの長さ）
problem_size = PIX_SIZE * PIX_SIZE

def make_blackbox_func(mnist_svm: MnistSvm) -> Callable[[np.ndarray], float]:
    def blackbox(x: np.ndarray) -> float:
        assert x.shape == (problem_size,)  # x は要素数 28*28 の一次元配列
        return -mnist_svm.train_eval(
            x
        )  # ブラックボックス関数として、モデルの正解率の負値を返却
    return blackbox

mnist_svm = MnistSvm(mask_ratio=0.75)

blackbox_func = make_blackbox_func(mnist_svm=mnist_svm)
# [True, True, ..., True, False, False, ... False] であるような
# マスク率 75% のマスクを与えた場合の機械学習モデルに対する正解率の負値
print(f"{blackbox_func(np.array([True] * 588 + [False] * 196)) = }")



# 乱数シードの固定
seed = 1234
rng = np.random.default_rng(seed)
torch.manual_seed(seed)

class TorchFM(nn.Module):
    def __init__(self, d: int, k: int):
        """モデルを構築する
        Args:
            d (int): 入力ベクトルのサイズ
            k (int): パラメータ k
        """
        super().__init__()
        self.d = d
        self.v = nn.Parameter(torch.randn((d, k)))
        self.w = nn.Parameter(torch.randn((d,)))
        self.w0 = nn.Parameter(torch.randn(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """入力 x を受け取って y の推定値を出力する
        Args:
            x (torch.Tensor): (データ数 × d) の 2 次元 tensor

        Returns:
            torch.Tensor: y の推定値 の 1次元 tensor (サイズはデータ数)
        """
        out_linear = torch.matmul(x, self.w) + self.w0
        out_1 = torch.matmul(x, self.v).pow(2).sum(1)
        out_2 = torch.matmul(x.pow(2), self.v.pow(2)).sum(1)
        out_quadratic = 0.5 * (out_1 - out_2)

        out = out_linear + out_quadratic
        return out

    def get_parameters(self) -> tuple[np.ndarray, np.ndarray, float]:
        """パラメータ v, w, w0 を出力する"""
        np_v = self.v.detach().numpy().copy()
        np_w = self.w.detach().numpy().copy()
        np_w0 = self.w0.detach().numpy().copy()
        return np_v, np_w, float(np_w0)



def compute_corrcoef(y0: torch.Tensor, y1: torch.Tensor):
    tens = torch.stack((y0, y1))
    corr_torch = torch.corrcoef(tens)
    return float(corr_torch[1, 0])

def train(x: np.ndarray, y: np.ndarray, model: TorchFM) -> float:
    """FM モデルの学習を行う
    Args:
        x (np.ndarray): 学習データ (入力ベクトル)
        y (np.ndarray): 学習データ (出力値)
        model (TorchFM): TorchFM モデル
    """
    # イテレーション数
    epochs = 2000
    # モデルの最適化関数
    optimizer = torch.optim.AdamW([model.v, model.w, model.w0], lr=0.3)
    # 学習率スケジューラ
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    # 損失関数
    loss_func = nn.MSELoss()
    # データセットの用意
    x_tensor, y_tensor = (
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float(),
    )
    dataset = TensorDataset(x_tensor, y_tensor)
    train_set, valid_set = random_split(dataset, [0.8, 0.2])
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    # 学習の実行
    min_loss = 1e18  # 損失関数の最小値を保存
    max_corrcoef = -1e18  # 相関係数の最大値を保存
    losses: list[float] = [0.0] * epochs
    corrcoefs: list[float] = [0.0] * epochs
    best_state = model.state_dict()  # 最も良いモデルのパラメータを保存するための変数
    for i in trange(epochs, leave=False):
        # 学習フェーズ
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            pred_y = model(x_train)
            loss = loss_func(pred_y, y_train)
            loss.backward()
            optimizer.step()

        # 検証フェーズ
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                out = model(x_valid)
                losses[i] += loss_func(out, y_valid)
            corrcoefs[i] = compute_corrcoef(model(x_tensor), y_tensor)
            if losses[i] < min_loss or corrcoefs[i] > max_corrcoef:
                # 損失関数又は相関係数の値が更新されたらパラメータを保存
                best_state = copy.deepcopy(model.state_dict())
                min_loss = losses[i]
                max_corrcoef = corrcoefs[i]
        scheduler.step()

    # モデルを学習済みパラメータで更新
    model.load_state_dict(best_state)
    return compute_corrcoef(model(x_tensor), y_tensor)



# ソルバークライアントを Amplify AE に設定
client = FixstarsClient()
client.parameters.timeout = timedelta(milliseconds=5000)
client.token = token  # ローカル環境等で実行する場合はコメントを外して Amplify AEのアクセストークンを入力してください

def anneal(
    torch_model: TorchFM,
    constraint: Callable[[PolyArray], Constraint],
) -> np.ndarray:
    """受け取った FM モデルの最小値を与える x を求める"""

    # 長さ d のバイナリ変数の配列を作成
    gen = VariableGenerator()
    x = gen.array("Binary", torch_model.d)

    # TorchFM からパラメータ v, w, w0 を取得
    v, w, w0 = torch_model.get_parameters()

    # 目的関数を作成
    out_linear = w0 + (x * w).sum()
    out_1 = ((x[:, np.newaxis] * v).sum(axis=0) ** 2).sum()  # type: ignore
    out_2 = ((x[:, np.newaxis] * v) ** 2).sum()
    objective: Poly = out_linear + (out_1 - out_2) / 2

    # 組合せ最適化モデルを構築
    amplify_model = Model(objective, constraint(x))

    # 最小化を実行（構築したモデルと、始めに作ったソルバークライアントを引数として渡す）
    result = solve(amplify_model, client)
    if len(result.solutions) == 0:
        raise RuntimeError(f"No solution was found.")

    # モデルを最小化する入力ベクトルを返却
    return x.evaluate(result.best.values).astype(int)



def init_training_data(
    n0: int, mnist_svm: MnistSvm, blackbox_func: Callable[[np.ndarray], float]
    ):
    """n0 組の初期教師データを作成する"""
    assert n0 < 2**problem_size

    # n0 個のマスクを乱数を用いて作成
    x = np.array([mnist_svm.generate_random_mask() for _ in range(n0)])

    # マスクの重複が発生していたらランダムに値を変更して回避する
    x = np.unique(x, axis=0)
    while x.shape[0] != n0:
        x = np.vstack((x, mnist_svm.generate_random_mask()))
        x = np.unique(x, axis=0)

    # blackbox 関数を評価して入力マスクに対応する n0 個の出力を得る
    y = np.zeros(n0) + 1e10
    for i in range(n0):
        start = time.perf_counter()
        y[i] = blackbox_func(x[i])
        print(
            f"Random process {i}: found y={y[i]:.4f}, current best={np.min(y):.4f}, "
            f"cycle elapsed time={time.perf_counter()-start:.2f}s"
        )
    return x, y



n0 = 10  # 初期教師データの数
x, y = init_training_data(n0, mnist_svm, blackbox_func)

# FMQA サイクルの実行回数
n = 20

def get_constraint(q: PolyArray) -> Constraint:
    """
    マスクされる特徴量は mnist_svm.num_masked_features 個という制約条件を
    返却する関数。制約条件の重みとして 2 を選択。制約条件の重みについては、
    https://amplify.fixstars.com/ja/docs/amplify/v1/constraint.html#id8 を参照。
    """
    return 2 * equal_to(q.sum(), mnist_svm.num_masked_features)

start_total = time.perf_counter()

# N 回のイテレーションを実行
for i in range(n):
    print(f"FMQA cycle {i}")
    start = time.perf_counter()
    # 機械学習モデルの作成
    model = TorchFM(problem_size, k=20)
    corrcoef = train(x, y, model)
    print(f"- corrcoef: {corrcoef:.2f}")
    print(f"- elapsed (train): {time.perf_counter() - start:.2f}s")

    # 学習済みモデルの最小値を与える入力ベクトルの値を取得
    start = time.perf_counter()
    x_hat = anneal(model, get_constraint)
    # x_hat が重複する場合、それに近い解を乱数に基づき生成する
    while (x_hat == x).all(axis=1).any():
        idx_0 = rng.choice(np.arange(problem_size))
        idx_1 = rng.choice(np.arange(problem_size))
        x_hat[idx_0], x_hat[idx_1] = x_hat[idx_1], x_hat[idx_0]
    print(f"- elapsed (anneal): {time.perf_counter() - start:.2f}s")

    # 推定された入力ベクトルを用いてブラックボックス関数を評価
    start = time.perf_counter()
    y_hat = blackbox_func(x_hat)
    print(f"- elapsed (eval): {time.perf_counter() - start:.2f}s")

    # 評価した値をデータセットに追加
    x = np.vstack((x, x_hat))
    y = np.append(y, y_hat)
    print(f"- found y = {y_hat:.4f}, current best = {np.min(y):.4f}")

print(f"total elapsed: {time.perf_counter() - start_total:.2f}s")



fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()
# 初期教師データ生成のブラックボックス関数の評価値
ax.plot(range(n0), y[:n0], marker="o", color="b")
# FMQA サイクルのブラックボックス関数の評価値
ax.plot(range(n0, n0 + n), y[n0:], marker="o", color="r")
# 最小値の更新履歴
ax.plot(range(n0 + n), [y[0]] + [min(y[:i]) for i in range(1, n0 + n)], color="k")
ax.set_xlabel("number of iterations", fontsize=18)
ax.set_ylabel("f(x)", fontsize=16)
ax.tick_params(labelsize=16)
plt.show()



def show_masked_images(
    mnist_svm: MnistSvm,
    num_rows: int = 2,
    num_cols: int = 4,
):
    data = mnist_svm._test_data
    labels = mnist_svm._test_labels
    assert data is not None and labels is not None
    _, test_data, _, _ = mnist_svm._fetch_masked_data()
    prediction = mnist_svm._model.predict(test_data)

    plt.figure(figsize=(10, 5))
    for i in range(min(num_rows * num_cols, len(data))):
        image, label = data.iloc[i].to_numpy(), labels.iloc[i]
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image.reshape(PIX_SIZE, PIX_SIZE), cmap="gray")
        for m, elm in enumerate(mnist_svm._mask):
            if elm:
                ax.add_patch(
                    patches.Rectangle(
                        xy=(m % PIX_SIZE - 0.5, int(m / PIX_SIZE) - 0.5),
                        width=0.6,
                        height=0.6,
                        color="red",
                        linewidth=None,
                    )
                )
        ax.axis("off")
        ax.set_title(f"{label} predicted as {prediction[i]}")
    plt.show()

show_masked_images(mnist_svm)


print('End')