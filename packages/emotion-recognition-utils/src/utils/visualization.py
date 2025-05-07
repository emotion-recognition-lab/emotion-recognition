from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


def collect_emotion_distribution_data(
    dataset_dir: Path,
    name_to_dataset: dict[str, type[Any]],
    available_split: set[str],
    emotion_mapping: dict[str, list[str]],
) -> pd.DataFrame:
    logger.info(f"支持的数据集: [blue]{name_to_dataset.keys()}[/]")
    logger.info(f"在 [blue]{dataset_dir}[/] 中搜索数据集")

    # 创建数据收集列表
    data_records = []
    split_order = {"train": 0, "valid": 1, "test": 2}  # 定义顺序
    split_names = {"train": "训练集", "valid": "验证集", "test": "测试集"}

    for name, dataset_cls in name_to_dataset.items():
        dataset_path = dataset_dir / name
        if not dataset_path.exists():
            continue

        logger.info(f"找到数据集: [blue]{name}[/]")
        emotions = emotion_mapping[name]

        for split in available_split:
            dataset = dataset_cls(dataset_path.as_posix(), split=split)
            logger.info(f"从 [blue]{name}[/] 加载 [blue]{split}[/] 数据集")

            labels = dataset.meta.apply(dataset.label2int, axis=1)
            emotion_counts = {emotions[i]: (labels == i).sum().__int__() for i in range(dataset.num_classes)}

            # 添加到数据记录列表
            for emotion, count in emotion_counts.items():
                data_records.append(
                    {
                        "Dataset": name,
                        "Split": split_names[split],
                        "Split_order": split_order[split],
                        "Emotion": emotion,
                        "Count": count,
                    }
                )

            logger.info(emotion_counts)

    df = pd.DataFrame(data_records)
    return df.sort_values("Split_order")


def plot_emotion_distribution(df: pd.DataFrame, output_dir: Path | None = None) -> None:
    # 设置中文字体
    plt.rcParams["font.family"] = ["Source Han Sans CN"]
    plt.rcParams["axes.unicode_minus"] = False

    # 获取唯一的数据集名称
    unique_datasets = df["Dataset"].unique()
    output_dir = output_dir or Path(".")

    # 为每个数据集创建独立的图形
    for dataset in unique_datasets:
        # 创建新的图形
        plt.figure(figsize=(20, 10))

        # 获取当前数据集的数据
        dataset_df = df[df["Dataset"] == dataset]
        ax = sns.barplot(
            data=dataset_df,
            x="Split",
            y="Count",
            hue="Emotion",
            order=["训练集", "验证集", "测试集"],
            width=0.9,
        )

        for rect in ax.patches:
            height = rect.get_height()  # type: ignore
            x = rect.get_x() + rect.get_width() / 2  # type: ignore
            y = height
            if height == 0:
                continue
            ax.annotate(
                f"{int(height)}",
                xy=(x, y),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=18,
            )

        plt.subplots_adjust(bottom=0.1, hspace=1.15)
        ax.set_xlabel("")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["训练集", "验证集", "测试集"], fontsize=30)
        ax.set_ylabel("\n".join(list("样本数量")), fontsize=35, rotation=0, labelpad=30)

        # 获取数据最大值并设置y轴范围
        max_value = dataset_df["Count"].max()
        y_max = max_value * 1.1  # 留出10%的空间
        ax.set_ylim(-max_value / 20, y_max)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_bounds(0, y_max)
        ax.spines["left"].set_bounds(0, y_max)
        ax.axhline(y=0, color="black", linewidth=1.0)
        ax.yaxis.set_tick_params(labelsize=20)

        # 设置标签样式
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
            label.set_verticalalignment("center")

        # 调整图例
        plt.legend(title="情感类别", bbox_to_anchor=(1, 1), loc="upper left", fontsize=30, title_fontsize=35)

        plt.tight_layout()
        output_path = output_dir / f"{dataset}_emotion_distribution.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        logger.info(f"已保存图表到: [blue]{output_path}[/]")


def plot_confusion_matrix(
    confusion_matrix: list[list[int]] | np.ndarray,
    class_names: list[str],
    output_path: Path | str = "confusion_matrix.png",
    normalize: bool = True,
    figsize: tuple[int, int] = (6, 4),
    fmt: str = ".2f",
    xlabel: str = "预测标签",
    ylabel: str = "真实标签",
    dpi: int = 300,
) -> None:
    plt.rcParams["font.family"] = ["Source Han Sans CN"]
    plt.rcParams["axes.unicode_minus"] = False

    cm = np.array(confusion_matrix)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_xlabel(xlabel, fontsize=15, rotation=0, labelpad=10)
    ax.set_ylabel("\n".join(list(ylabel)), fontsize=15, rotation=0, labelpad=10)
    ax.yaxis.set_label_coords(-0.25, 0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    logger.info(f"Saved confusion matrix to: [blue]{output_path}[/]")
