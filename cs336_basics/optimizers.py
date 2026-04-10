import math
from typing import Callable, Optional, Dict, Any
import torch
from torch import nn
from torch.optim.optimizer import ParamsT

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr=1e-3, betas=(0.9, 0.95), weight_decay=0, eps=1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invaild learning rate: {lr}")
        defaults = {"lr": lr, "m": 0, "v": 0, "lamda": weight_decay, "eps": eps, "betas": betas}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None) -> Callable | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lamda = group["lamda"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 1)   # 获取迭代次数，注意 t 是从 1 开始
                m = state.get("m", torch.zeros_like(p))   # 获取一阶动量
                v = state.get("v", torch.zeros_like(p))   # 获取二阶动量
                # grad = p.grad.data
                grad = p.grad
                # 更新动量和学习率
                # m = beta1 * m + (1 - beta1) * grad
                # v = beta2 * v + (1 - beta2) * (grad ** 2)
                # 原地更新
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                # 更新参数
                # p.data -= lr_t * m / (torch.sqrt(v) + eps) + lr * lamda * p.data
                with torch.no_grad():
                    p.mul_(1 - lr * lamda)
                    denom = v.sqrt().add_(eps)
                    p.addcdiv_(m, denom, value=-lr_t)
                # 更新状态
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        
        return loss

class SGD(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invaild learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None) -> Callable | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                t = state.get("t", 0)   # 迭代次数
                grad = p.grad.data      # 梯度数据
                p.data -= lr / math.sqrt(t + 1) * grad  # SGD更新参数公式
                state["t"] = t + 1
        
        return loss


def test_sgd():
    weights = nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e1)

    for _ in range(100):
        opt.zero_grad()     # 清空梯度
        loss = (weights ** 2).mean()    # 计算损失标量
        print(loss.item())
        loss.backward()     # 反向传播计算梯度
        opt.step()          # 更新参数


def test_adamw():
    weights = nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1e1)

    for _ in range(100):
        opt.zero_grad()     # 清空梯度
        loss = (weights ** 2).mean()    # 计算损失标量
        print(loss.item())
        loss.backward()     # 反向传播计算梯度
        opt.step()          # 更新参数

if __name__ == "__main__":
    test_adamw()