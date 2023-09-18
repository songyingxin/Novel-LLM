

## 文件架构
- baichuan + run_one.py： baichuan的原始代码，可以用来调试
- attention_mask: attention_mask 的转换函数
- baichuan-7b: baichuan 的核心代码逻辑
- rotary_embedding.py: 旋转位置编码的逻辑

## 调试

1. 更改 run_one.py 中的模型路径
2. 开始调试 python -m pdb run_one.py