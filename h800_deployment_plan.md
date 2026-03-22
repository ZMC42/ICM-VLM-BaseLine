# H800 上部署 Qwen3.5 的当前方案说明

## 1. 目标与约束

当前方案的目标是在 **H800** 上使用本地修补后的 `vLLM` 部署并推理 `Qwen3.5` 系列模型，同时规避已定位的 Hopper 专用内核兼容问题。

当前已知硬约束：

- H800 所在机器的驱动环境上限为 **CUDA 12.6**。
- `vLLM 0.17.1` 的官方 **precompiled wheel** 不提供可直接用于当前约束的 `cu126` 变体。
- `Qwen3.5` 在 `Hopper (SM90)` 上，`vLLM 0.17.1` 默认会在 `qwen3_next.py` 中走 `FlashInfer GDN prefill` 路径。
- 上述路径在你的环境中已经实际触发：
  - `Failed to initialize the TMA descriptor 999`
- 因此当前方案不再依赖 precompiled wheel，而是改为：
  - **本地源码修补**
  - **H800 上完整源码编译 vLLM**

---

## 2. 已定位的问题与处理策略

### 2.1 Hopper 上的 GDN prefill / TMA 崩溃

在 `vLLM 0.17.1` 的 `releases/v0.17.1` 分支中：

- `vllm/model_executor/models/qwen3_next.py` 对 `SM90/Hopper` 是硬编码行为：
  - 默认走 `self.forward_cuda`
  - 日志表现为：`Using FlashInfer GDN prefill kernel on CUDA compute capability 90`
- 该路径在你的 H800 环境中会触发 `TMA descriptor 999` 失败。

### 2.2 当前修复方式

已在本地 `dependencies/vllm` 中提交 patch：

- commit: `77129a906`
- commit message: `Disable Hopper FlashInfer GDN prefill`

该 patch 的作用：

- 对 `SM90/Hopper` 不再走 `forward_cuda`
- 改为强制走 `forward_native`
- 从而绕开 `FlashInfer GDN prefill -> TMA` 这条失败路径

### 2.3 为什么不再使用 precompiled wheel

原因不是“不想用”，而是 **当前版本的 wheel 变体不匹配你的驱动约束**。

已确认事实：

- `vLLM 0.17.1` 的 precompiled wheel 逻辑对 CUDA 12.x 会映射到 `cu129`
- 不存在可直接使用的 `cu126` metadata / wheel 路径
- 对 H800 当前“驱动最高 12.6”的约束，不能安全地退回 `cu129`

所以当前可行路线变为：

- 直接在 H800 上从源码编译 `vLLM`
- 使用本地 patch 后的 `dependencies/vllm`
- 使用 `cu126` 的 PyTorch 运行时栈

---

## 3. 现有脚本说明

### 3.1 `h800_setup.sh`

文件：`h800_setup.sh`

职责：

1. 创建/复用 conda 环境 `ICM-VLM`
2. 安装源码编译所需工具：
   - `pip`
   - `setuptools`
   - `wheel`
   - `packaging`
   - `numpy`
   - `cmake`
   - `ninja`
3. 优先从本地离线 wheel 目录 `torch_cu126_wheels/` 安装依赖；若不存在，再回退到在线安装
4. 安装 `cu126` 的 PyTorch 栈：
   - `torch==2.10.0`
   - `torchvision==0.25.0`
   - `torchaudio==2.10.0`
5. 安装调用服务需要的 Python 包：
   - `openai`
   - `httpx`
6. 做编译前自检：
   - `CUDA_HOME/bin/nvcc`
   - `gcc`
   - `g++`
   - `nvidia-smi`
   - 打印 `nvcc --version`
   - 打印 `nvidia-smi`
7. 设置源码编译相关环境变量：
   - `CUDA_HOME`
   - `PATH`
   - `LD_LIBRARY_PATH`
   - `MAX_JOBS`
   - `NVCC_THREADS`
   - `VLLM_USE_PRECOMPILED=0`
8. 对 `dependencies/vllm` 执行：
   - `pip install -e ... --no-build-isolation`

### 3.2 `h800_server.sh`

文件：`h800_server.sh`

职责：

- 使用 H800 上的 patched/source-built `vLLM` 启动正式服务
- 默认模型是：`ModelHub/Qwen/Qwen3.5-27B`

关键启动参数：

- `--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":2}'`
  - 强制使用 `FlashAttention v2`
- `--enforce-eager`
  - 避免 CUDA graph / compile 路径干扰
- `--language-model-only`
  - 只走文本链路，减少不必要的多模态分支干扰
- `--reasoning-parser qwen3`
  - 与 Qwen3.5 的推理输出格式对齐

### 3.3 `h800_smoke_test.sh`

文件：`h800_smoke_test.sh`

职责：

- 在 H800 上先验证 **小模型 + 基础推理链路**，再决定是否跑 `Qwen3.5-27B`

默认行为：

- 使用模型：`ModelHub/Qwen/Qwen3.5-9B`
- 端口：`18000`
- `max-model-len=4096`
- `gpu-memory-utilization=0.85`

测试流程：

1. 打印当前 `torch` / `cuda(runtime)` / `gpu` / `vllm`
2. 后台拉起 vLLM 服务
3. 轮询 `/v1/models` 等待服务 ready
4. 发送一次最小 `chat/completions` 请求
5. 检查是否返回 `content` 或 `reasoning`
6. 自动清理后台进程

它的意义不是替代正式部署，而是尽快回答两个问题：

- 这台 H800 上的 `vLLM` 源码编译结果是否可运行
- 小模型推理链路是否通畅

---

## 4. 推荐执行顺序

### 第一步：确认 H800 机器满足源码编译前提

至少要满足：

- `CUDA_HOME` 指向本机 **CUDA 12.6 toolkit**
- `nvcc --version` 可运行
- `gcc` / `g++` 可用
- `nvidia-smi` 可用

示例：

```bash
export CUDA_HOME=/usr/local/cuda-12.6
${CUDA_HOME}/bin/nvcc --version
nvidia-smi
```

### 第二步：执行环境安装与源码编译

如果 H800 机器可以访问外网中的 PyTorch 官方 `cu126` 仓库，则直接执行：

```bash
CUDA_HOME=/usr/local/cuda-12.6 bash h800_setup.sh
```

如果 H800 机器的镜像源不包含 `torch 2.10.0+cu126`，则先在有外网的机器上准备离线包：

```bash
bash prepare_torch_cu126_offline.sh
```

把生成的 `torch_cu126_wheels/` 目录拷到 H800 后，再执行：

```bash
WHEEL_DIR=/path/to/torch_cu126_wheels CUDA_HOME=/usr/local/cuda-12.6 bash h800_setup.sh
```

说明：

- 如果你的 toolkit 不在 `/usr/local/cuda-12.6`，替换成真实路径
- `h800_setup.sh` 会优先使用本地 wheel；若本地目录不存在，再回退到在线安装
- 该步骤会直接从源码编译 `vLLM`

### 第三步：先做冒烟测试

```bash
bash h800_smoke_test.sh
```

若成功，说明：

- 当前 patched `vLLM` 可以至少完成：
  - 启动服务
  - 加载小模型
  - 完成一个最小推理请求

### 第四步：再启动正式 27B 服务

```bash
bash h800_server.sh
```

---

## 5. 为什么先测 9B，再跑 27B

当前方案虽然已经规避了最明确的 `GDN prefill / TMA` 失败点，但仍然不能逻辑上保证：

- H800 上 `Qwen3.5-27B` 一定无其它 Hopper 特化路径问题
- H800 上源码编译得到的 `vLLM` 一定不存在别的运行时问题

所以更合理的工程顺序是：

1. 先验证编译是否成功
2. 先验证小模型是否能完成完整推理链路
3. 再把问题收敛到 `27B` 模型规模本身

这样出错时更容易判断：

- 是编译问题
- 是基础部署问题
- 还是 `Qwen3.5-27B` 自身在 Hopper 上仍有特殊兼容性问题

---

## 6. 当前方案的边界

### 已覆盖的问题

- `vLLM 0.17.1` 无 `cu126` precompiled wheel
- Hopper 上 `FlashInfer GDN prefill` 导致的 `TMA descriptor 999`
- `FA3` / `CUDA graph` 相关的额外干扰

### 尚不能保证的部分

当前方案是“高概率可行”，但不是“确定成功保证”。

仍存在的不确定性包括：

- H800 本机 `CUDA 12.6` toolchain 与 `torch cu126` 的具体兼容细节
- 本机 `gcc/glibc/nccl/cmake/ninja` 版本差异
- `Qwen3.5-27B` 在 Hopper 上可能存在的其他专用 kernel 路径问题
- 大模型在实际显存占用、并发、上下文长度配置下的稳定性问题

因此正确的表述应该是：

- 这是在你当前约束下 **最合理、最可辩护** 的方案
- 它不是数学意义上的“必然成功”方案
- 所以才需要 `h800_smoke_test.sh`

---

## 7. 关键文件清单

- `dependencies/vllm`
  - 本地 `vLLM` 源码
  - 当前使用分支：`releases/v0.17.1`
  - 包含 patch commit：`77129a906`

- `h800_setup.sh`
  - H800 上源码编译安装脚本
  - 支持离线优先安装 `torch 2.10.0+cu126`

- `prepare_torch_cu126_offline.sh`
  - 在有外网的机器上准备 `torch 2.10.0+cu126` 及相关依赖的离线 wheel 包

- `h800_server.sh`
  - H800 上正式服务启动脚本

- `h800_smoke_test.sh`
  - H800 上小模型冒烟测试脚本

---

## 8. 建议的实际操作命令

### 安装

```bash
CUDA_HOME=/usr/local/cuda-12.6 bash h800_setup.sh
```

### 冒烟测试

```bash
bash h800_smoke_test.sh
```

### 正式启动

```bash
bash h800_server.sh
```

---

## 9. 总结

当前 H800 方案的核心思想是：

- 不再尝试使用与 `CUDA 12.6` 不匹配的 precompiled wheel
- 保留你已经验证有效的本地 patch
- 在 H800 上直接源码编译 `vLLM 0.17.1`
- 先用 `9B` 验证部署链路，再上线 `27B`

这是目前在你给定约束下，最稳妥、最可维护的实施路径。
