import math
from typing import Callable, Optional, Dict, Any
import torch
from torch import nn
from torch.optim.optimizer import ParamsT

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr=1e-3, betas=(0.9, 0.95), weight_decay=0, eps=1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invaild learning rate: {lr}")
        defaults = {"lr": lr, "m": 0, "v": 0}
        super().__init__(params, defaults)

        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.lamda = weight_decay
        self.eps = eps

    def step(self, closure: Optional[Callable]=None) -> Callable | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)   # 获取迭代次数
                m = state.get("m", 0)   # 获取一阶动量
                v = state.get("v", 0)   # 获取二阶动量
                grad = p.grad.data
                # 更新动量和学习率
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
                lr_t = lr * math.sqrt((1 - self.beta2) ** t) / ((1 - self.beta1) ** t)
                # 更新参数
                p.data -= lr_t * m / (torch.sqrt(v) + self.eps)
                p.data -= lr * self.lamda * p.data
                # 更新状态
                self.state["t"] = t + 1
                self.state["m"] = m
                self.state["v"] = v
        
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