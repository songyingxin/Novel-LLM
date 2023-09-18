
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PathArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        default="",
        metadata={"help": "预训练模型的路径"}
    )

    train_file: Optional[str] = field(
        default="", 
        metadata={"help": "训练集数据"}
    )
    validation_file: Optional[str] = field(
        default="",
        metadata={"help": ("验证集数据")},
    )
    output_dir: Optional[str] = field(
        default="./output_finetune",
        metadata={"help": "模型输出文件夹路径"},
    )
    output_model_name: Optional[str] = field(
        default="baichuan",
        metadata={"help": "控制日志和输出路径"},
    )
    logging_dir: Optional[str] = field(
        default="./log_finetune",
        metadata={"help": "日志输出文件夹路径"},
    )

@dataclass
class TrainerArguments:
    """
    训练相关参数
    """
    max_length: Optional[int] = field(
        default=64,
        metadata={"help": ("输入 prompt 的最大长度")},
    )
    seed: Optional[int] = field(
        default=1234,
        metadata={"help": ("种子")},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "deepspeed 文件路径"},
    )
    local_rank: Optional[int] = field(
        default=-1,
        metadata={"help": ("local rank")},
    )
    learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={"help": "学习率"},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": ("单个GPU的 train batch size")},
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": ("单个GPU的 eval batch size")},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": ("梯度累积")},
    )
    num_train_epochs: Optional[int] = field(
        default=5,
        metadata={"help": ("训练的 epoch 数目")},
    )
    save_step: Optional[int] = field(
        default=500,
        metadata={"help": ("多少步进行一次eval，日志打印以及保存模型")},
    )



    