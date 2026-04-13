import wandb, csv
from loguru import logger
from pathlib import Path
import numpy as np
import time

from cs336_basics.configs import Config
from cs336_basics.transformer import Transformer
from cs336_basics.optimizers import AdamW
from cs336_basics.training_utils import (
    load_checkpoint, 
    save_checkpoint, 
    get_batch, 
    gradient_clipping, 
    cosine_learning_rate_schedule, 
    cross_entropy,
)


def main_pipeline():
    logger.info("获取配置文件")
    config_file = Path(__file__).parent / "config.yaml"
    config = Config.from_yaml(config_file)

    logger.info("初始化模型、优化器")
    # 初始化模型、优化器
    model = Transformer.from_config(config)
    model.to(config.device)
    model.train()
    optim = AdamW.from_config(model.parameters(), config)

    # checkpoint确认当前训练循环
    if config.is_checkpoint:
        cur_iters = load_checkpoint(config.checkpoint_path, model, optim)
    else:
        cur_iters = 0
    
    total_iters = config.train.total_train_iters
    assert cur_iters < total_iters

    logger.info("加载训练数据集")
    # 获取数据集
    train_dataset = np.memmap(config.train.train_data, dtype=np.uint16)

    wandb.init(
        project="cs336-hw1",
        config=config.to_dict()
    )
    output_dir = Path(config.train.output_dir)
    if not Path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(output_dir / "config.yaml")
    # 创建csv写入部分
    csv_path = output_dir / "train_log.csv"
    csv_headers = ["iteration", "loss", "lr", "batch_per_sec", "token_per_sec", "total_tokens"]
    if not config.is_checkpoint or not csv_path.exists():
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)

    start = time.time()
    dura_batch = 0
    dura_token = 0
    total_tokens = 0
    for itera in range(cur_iters + 1, total_iters + 1):
        # 计算本轮更新的学习率
        lr = cosine_learning_rate_schedule(
            itera,
            config.optim.max_lr,
            config.optim.min_lr,
            config.train.warmup_iters,
            config.train.cosine_iters
        )
        for group in optim.param_groups:
            group["lr"] = lr

        x_batch, y_batch = get_batch(
            train_dataset,
            batch_size=config.train.batch_size,
            context_length=config.model.context_length,
            device=config.device,
        )
        # 清空优化器梯度
        optim.zero_grad()
        # 计算loss，反向传播计算梯度
        logits = model(x_batch)
        loss = cross_entropy(logits, y_batch)
        loss.backward()
        # 对计算出来的梯度进行裁剪
        gradient_clipping(model.parameters(), config.train.max_l2_norm)
        # 更新梯度
        optim.step()

        # 更新统计信息
        dura_batch += x_batch.size()[0]
        dura_token += x_batch.numel()
        total_tokens += x_batch.numel()

        # 打印信息
        if itera % config.train.log_interval == 0:
            duration = time.time() - start
            batch_per_sec = dura_batch / duration
            token_per_sec = dura_token / duration
            dura_batch = 0
            dura_token = 0

            wandb.log({
                "train/loss": loss.item(),
                "train/iteration": itera,
                "train/learning_rate": lr,
                "train/batch_per_sec": batch_per_sec,
                "train/token_per_sec": token_per_sec,
                "train/total_tokens": total_tokens
            })
            # 写入csv
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    itera, 
                    f"{loss.item():.5f}", 
                    f"{lr:.8f}", 
                    f"{batch_per_sec:.2f}", 
                    f"{token_per_sec:.2f}", 
                    total_tokens
                ])
            logger.info(f"Iteration: {itera}, train loss: {loss.item():.5f}")
            start = time.time()
        
        # 保存checkpoint
        if itera % config.train.ckpt_interval == 0:
            save_path = output_dir / f"checkpoint_{itera}.pt"
            save_checkpoint(model, optim, itera, save_path)
        
    wandb.finish()
    logger.info("=" * 10 + f" Training complete, final loss: {loss.item():.5f} " + "=" * 10)


if __name__ == "__main__":
    main_pipeline()