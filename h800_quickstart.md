# H800 Quickstart

## 1. 准备 Torch 离线包（如果 H800 镜像源没有 `torch 2.10.0+cu126`）

```bash
bash prepare_torch_cu126_offline.sh
```

把生成的 `torch_cu126_wheels/` 目录拷到 H800 机器。

## 2. 在 H800 上编译安装 patched vLLM

在线安装 Torch：

```bash
CUDA_HOME=/usr/local/cuda-12.6 bash h800_setup.sh
```

离线优先安装 Torch：

```bash
WHEEL_DIR=/path/to/torch_cu126_wheels CUDA_HOME=/usr/local/cuda-12.6 bash h800_setup.sh
```

## 3. 先跑小模型冒烟测试

```bash
bash h800_smoke_test.sh
```

## 4. 再启动正式 27B 服务

```bash
bash h800_server.sh
```

## 5. 关键要求

- `dependencies/vllm` 必须包含 patch commit `77129a906`
- H800 本机必须有可用的 CUDA 12.6 toolkit
- `h800_setup.sh` 安装后会校验：
  - `torch.__version__` 包含 `+cu126`
  - `torch.version.cuda == 12.6`
