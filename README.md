# ai-for-cfd-notebooks

物理信息机器学习（PINN）与神经算子（FNO）在计算流体力学中的应用实验。

## 内容

| Notebook | 方法 | 问题 |
|----------|------|------|
| `pinn_burgers.py` | PINN | 1D Burgers 方程，与 WENO5 对比 |
| `pinn_heat.py` | PINN | 1D 热传导方程，多工况泛化测试 |

## 环境

```bash
pip install torch numpy matplotlib scipy
```

PyTorch >= 2.0，CPU 可运行，GPU 可选。

## 参考文献

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *J. Comput. Phys.* 378, 686–707.
- Li, Z. et al. (2021). Fourier neural operator for parametric partial differential equations. *ICLR 2021*.
