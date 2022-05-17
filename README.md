<center><h1>Bootstrapped MAE (MAE-K) Pytorch实现</h1></center>

#### 代码修改自：https://github.com/liujiyuan13/MAE-code

#### 运行准备：

1. 克隆代码到本地仓库

```bash
git clone https://github.com/zhangkai0425/MAE-K.git
```

2. 配置相应环境   `cuda:11.0|NVIDIA GeForce RTX 3090` 

```bash
cd mae-k
cd env
conda env create -f environment.yaml
conda activate SurRF
```

3. 数据集准备

   本实验中使用cifar-10数据集，调用`torchvision.dataset.CIFAR10`，自动下载，无需自己准备

4. 设置参数

   所有的参数均在 `main_mae(eval).py` 文件下的`default_args()` 中，需要手动调整，沿用了`Github`源代码，没有封装成`argpars`命令行的形式

5. 任务目前采用单`GPU`训练的方式，由于数据集比较小，没有改成多核训练的方式，但可以通过`CUDA_VISIBLE_DEVICES`命令在多个`GPU`上运行不同的任务

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
2. 对于`MAE` **Linear Evaluation**，设置`main_eval.py`参数及主函数如下：

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

3. 对于`Bootstrapped MAE` **Linear Evaluation**，设置`main_eval.py`参数及主函数如下：

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
2. 对于`MAE` **Finetune**，设置`main_eval.py`参数及主函数如下：

`args.n_partial = 1`

```python
if __name__ == '__main__':
    data_name = 'cifar10'
    trail = 'MAE'
    train(default_args(data_name,trail=trail,ckpt_file='MAE.ckpt'))
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_eval.py | tee -a log_finetune/log_MAE_Finetune.txt   # Finetune
```

3. 对于`Bootstrapped MAE`**Finetune**，设置`main_eval.py`参数及主函数如下：

`args.n_partial = 1`

```python
if __name__ == '__main__':
    data_name = 'cifar10'
    trail = 'MAE-n'
    train(default_args(data_name,trail=trail,ckpt_file='MAE-n.ckpt'))
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python main_eval.py | tee -a log_finetune/log_MAE_n_Finetune.txt  # Finetune
```

#### 文件清单

```bash
...
+ ckpt              # checkpoint
+ data              # data folder
+ log               # log files of tensor board(the original log file)
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

1. 可视化`pretrain`阶段结果：

```bash
cd log_pretraining
python plot_pretraining.py
```

2. 可视化`linear evaluation`结果

```bash
cd log_linear
python plot_linear_evaluation.py
```

3. 可视化`finetune`结果

```bash
cd log_finetune
python plot_finetune.py
```
