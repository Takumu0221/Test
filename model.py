from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as f
from torchsummary import summary
import pytorch_lightning as pl
from torch.utils import data
from torch.utils.data import dataset, sampler, BatchSampler, DataLoader
import torchvision
import os

from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Dataset(dataset.Dataset):
    ## 本当は自作クラスを使わなくてもいいんだけど、一応作ります(自作メソッド足したくなるかも。画像表示とか。)
    ## self.mnist_dataとselfの、継承元は同じクラスなので、self.mnist_dataをそのままdataloaderに突っ込んでも大丈夫
    def __init__(self, train=True, transform=lambda x: x):
        if os.path.exists("./datas/MNIST/processed/training.pt") and os.path.exists("./datas/MNIST/processed/test.pt"):
            mnist_data = torchvision.datasets.MNIST("./datas/", train=train, download=False, transform=transform)
        else:
            mnist_data = torchvision.datasets.MNIST("./datas", train=train, download=True, transform=transform)
        self.mnist_data = mnist_data
        self.transform = transform

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        item = self.mnist_data[idx]
        return item[0].squeeze(), item[1]
        # return self.transform(self.mnist_data[idx])


class Transform():
    def __init__(self):
        # 使いたければ使ってください。
        # 自作Transform pytorchで検索
        pass

    def __call__(self, x):
        return x


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.data_dir = "./datas" # めんどいのでdata置き場へのpathは各所で手書きした。改良よろ
        # 前処理の部分は調べて「なんとか変換」するように変更してもOK
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))  # この数字はテキトーにした
        ])  # デフォルトの前処理。これの代わりにTransformクラスを実装してTransform.__call__()メソッドを使えるようにしてもいい

        self.batch_size = 32
        self.eps = 1e-10

        input_num = 28 * 28
        hidden = 10 * 50
        self.l1 = torch.nn.Linear(int(25600 * 4 / 32), hidden)  # 一層の線形層
        self.l2 = torch.nn.Linear(hidden, 10)  # 第2層
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=(5, 5))  # 畳み込み層1
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=(5, 5))  # 畳み込み層2
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.dropout1 = torch.nn.Dropout2d(0.25)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.relu = torch.nn.ReLU()

    def criterion(self, y_hat, y):
        return -torch.log(torch.gather(y_hat, -1, y.unsqueeze(1)) + self.eps)

    def forward(self, x):  # def __call__(self.x)
        # ソフトマックス関数に通す！　&　追加した層への入力を行う（線形層，畳み込み層）
        # x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, int(25600 * 4 / 32))
        x = torch.relu(self.l1(x))
        x = self.dropout2(x)
        x = self.l2(x)
        # return self.softmax(x)  # 一層の線形層に入力して活性化層に入力
        return torch.softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.mean(self.criterion(y_hat, y))  # 損失関数に入れる
        preds = torch.topk(y_hat, 1)[1].squeeze()
        acc = len(torch.where((y - preds) == 0)[0]) / len(y)  # 精度
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {"loss": loss, "acc": acc}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        acc = [outputs[i]["acc"] for i in range(len(outputs))]
        avg_acc = sum(acc) / len(acc)
        avg_loss = torch.stack([outputs[i]["loss"] for i in range(len(outputs))]).mean()
        self.log("train/epoch_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/epoch_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # 交差検証用に使う（モデルを常に未知のデータで評価する）
        x, y = batch
        y_hat = self(x)
        loss = torch.mean(self.criterion(y_hat, y))
        preds = torch.topk(y_hat, 1)[1].squeeze()
        acc = len(torch.where((y - preds) == 0)[0]) / len(y)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        acc = [outputs[i]["acc"] for i in range(len(outputs))]
        avg_acc = sum(acc) / len(acc)
        avg_loss = torch.stack([outputs[i]["loss"] for i in range(len(outputs))]).mean()
        self.log("val/epoch_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/epoch_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, bath_idx):
        # テストデータでの検証
        # 訓練ステップで変更を加えた場合，Validation stepとTest stepにも変更を加える
        x, y = batch
        y_hat = self(x)
        loss = torch.mean(self.criterion(y_hat, y))
        preds = torch.topk(y_hat, 1)[1].squeeze()
        acc = len(torch.where((y - preds) == 0)[0]) / len(y)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "acc": acc}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        acc = [outputs[i]["acc"] for i in range(len(outputs))]
        avg_acc = sum(acc) / len(acc)
        avg_loss = torch.stack([outputs[i]["loss"] for i in range(len(outputs))]).mean()
        self.log("test/epoch_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test/epoch_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # ここを変えてもOK（Optimizerの種類，パラメタ）
        return torch.optim.Adam(self.parameters(), lr=0.02)
        # return torch.optim.SGD(self.parameters(), lr=0.01, momentum=5)

    # 以下の各種dataloader,prepare_data,setupのメソッドは本来pl.LightningDataModuleとしてまとめるんだけど(同様のデータを使う異なるモデルに転用するため)
    # 今回は他のモデルに転用予定がないしめんどくさいのでpl.LightningModuleとまとめちゃいます
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def prepare_data(self) -> None:
        # Downloadしたりとかする。マルチGPUに分割される前に呼び出される Download済みか確認してされてなければDownloadだけする Use this method to do things that
        # might write to disk or that need to be done only from a single process in distributed settings.
        if os.path.exists("./datas/MNIST/processed/training.pt") and os.path.exists("./datas/MNIST/processed/test.pt"):
            pass
        else:
            torchvision.datasets.MNIST("./datas", download=True)

    def setup(self, stage: Optional[str] = None):
        # stageには"fit"か"test"が入っている。Noneが入っているならそれは手動でこの関数を呼び出している(e.g.デバック時)
        """
            There are also data operations you might want to perform on every GPU. Use setup to do things like:

            count number of classes
            build vocabulary
            perform train/val/test splits
            apply transforms (defined explicitly in your datamodule or assigned in init)
            etc…
        """
        if stage in (None, 'fit'):
            # 訓練時に呼び出される
            full_dataset = Dataset(train=True,
                                   transform=self.transform
                                   )
            self.mnist_train, self.mnist_val = dataset.random_split(full_dataset,
                                                                    [len(full_dataset) - int(len(full_dataset) * 0.1),
                                                                     int(len(full_dataset) * 0.1)])
            self.dims = tuple(self.mnist_train[0][0].shape)
        if stage in (None, "test"):
            # テスト時に呼び出される
            self.mnist_test = Dataset(train=False, transform=self.transform)
            self.dims = tuple(self.mnist_train[0][0].shape)


if __name__ == "__main__":
    logger = TensorBoardLogger('tb_logs', name="mnist")  # 文字列は適宜変えて
    torch.manual_seed(0)  # seedを固定して再現性を確保
    model = Model()
    print(summary(model, (28, 28)))
    trainer = pl.Trainer(
        max_epochs=10,  # 最大何エポック回すか
        logger=logger,  # tensorboardloggerを使用して,lossの推移を観察
        # callbacks=[EarlyStopping(monitor='v/avg_loss')],#早期終了,使うならググって条件を変えてください
        # gradient_clip_val=2.0 #勾配クリッピング,数字はいい感じでよろ
    )
    trainer.fit(model)  # 学習,結果は自動で保存される
    trainer.test(model)  # テスト,テストデータで性能を評価
