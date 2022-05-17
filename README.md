# Bootstrapped MAE (MAE-K) Pytorch实现

#### 代码修改自：https://github.com/liujiyuan13/MAE-code

#### 运行准备：

1. 克隆代码到本地仓库

```bash
git clone https://github.com/liujiyuan13/MAE-code.git MAE-code
```

2. 配置相应环境   `cuda:11.0|NVIDIA GeForce RTX 3090` 

```bash
cd mae-k
cd env
conda env create -f environment.yaml
conda activate SurRF
```

3. 数据集准备

​		本实验中使用cifar-10数据集，调用`torchvision.dataset.CIFAR10`，自动下载，无需自己准备

4. 设置参数

   所有的参数均在 `main_mae(eval).py` 文件下的`default_args()` 中，需要手动调整，沿用了`Github`源代码，没有封装成`argpars`命令行的形式

#### MAE/Bootstrapped MAE 预训练

1. 进入根目录下
2. 对于`MAE`预训练，设置`main_mae.py`主函数如下：

```python
if __name__ == '__main__':
    data_name = 'cifar10'
    trail = 'MAE'
    # mode 'MAE-K' or 'MAE'
    mode = 'MAE'
    if mode == 'MAE':
        train(default_args(data_name,trail=trail))
    if mode == 'MAE-K':
        train_k(default_args(data_name,trail=trail))
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_mae.py | tee -a log_pretraining/log_MAE.txt    # train MAE encoder
```

3. 对于`Bootstrapped MAE`预训练，设置`main_mae.py`主函数如下：

```bash
if __name__ == '__main__':
    data_name = 'cifar10'
    trail = 'MAE-n' # n是想训练的MAE阶段次数
    # mode 'MAE-K' or 'MAE'
    mode = 'MAE-K'
    if mode == 'MAE':
        train(default_args(data_name,trail=trail))
    if mode == 'MAE-K':
        train_k(default_args(data_name,trail=trail))
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_mae.py | tee -a log_pretraining/log_MAE_n.txt    # train MAE encoder
```

#### MAE/Bootstrapped MAE  Linear Evaluation

1. 进入根目录下
2. 对于`MAE` **Linear Evaluation**，设置`main_eval.py`主函数参数如下：

`args.n_partial = 0`

```python
if __name__ == '__main__':
    data_name = 'cifar10'
    trail = 'MAE'
    train(default_args(data_name,trail=trail,ckpt_file='MAE.ckpt'))
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_eval.py | tee -a log_linear/log_MAE_Linear.txt   # Linear Evaluation
```

3. 对于`Bootstrapped MAE` **Linear Evaluation**，设置`main_eval.py`主函数如下：

`args.n_partial = 0`

```python
if __name__ == '__main__':
    data_name = 'cifar10'
    trail = 'MAE-n'
    train(default_args(data_name,trail=trail,ckpt_file='MAE-n.ckpt'))
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_eval.py | tee -a log_linear/log_MAE_n_Linear.txt  # Linear Evaluation
```

#### MAE/Bootstrapped MAE Finetune

1. 进入根目录下
2. 对于`MAE` **Finetune**，设置`main_eval.py`主函数如下：

`args.n_partial = 0`

```python
if __name__ == '__main__':
    data_name = 'cifar10'
    trail = 'MAE'
    train(default_args(data_name,trail=trail,ckpt_file='MAE.ckpt'))
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_eval.py | tee -a log_linear/log_MAE_Linear.txt   # Linear Evaluation
```

3. 对于`Bootstrapped MAE`**Finetune**，设置`main_eval.py`主函数如下：

`args.n_partial = 0`

```python
if __name__ == '__main__':
    data_name = 'cifar10'
    trail = 'MAE'
    train(default_args(data_name,trail=trail,ckpt_file='MAE.ckpt'))
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_eval.py | tee -a log_linear/log_MAE_Linear.txt   # Linear Evaluation
```

#### 文件清单

```bash
...
+ ckpt              # checkpoint
+ data              # data folder
+ log               # log files
+ log_pretraining   # log pretraining 
+ log_linear        # log linear evaluation
+ log_finetune      # log whole network finetuning
README.md 
lars.py             # LARS optimizer
main_eval.py        # main file for linear evaluation/whole network finetuning
main_mae.py         # main file for MAE/Bootstrapped MAE  pretraining
model.py            # model definitions of MAE and EvalNet(MAE + linear network)
util.py             # helper functions
vit.py              # definition of vision transformer
deit.py             # definition of DeiT
```


#### 可视化结果

