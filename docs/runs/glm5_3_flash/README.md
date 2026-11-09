# GLM-5.3-Flash

## 命名

代码目录和文件统一使用 `glm5_3_flash`：

- 模型目录：`src/torchturbo/models/glm5_3_flash/`
- 配置文件：`configuration_glm5_3_flash.py`
- 模型文件：`modeling_glm5_3_flash.py`
- 回归测试：`tests/models/glm5_3_flash/`

类名保留 `Glm5NextConfig`、`Glm5NextForConditionalGeneration` 和
`TurboGlm5NextForConditionalGeneration` 等现有名称。模型注册名和权重配置中的
`model_type` 仍为 `glm5_next`（子配置为 `glm5_next_text`、`glm5_next_vision`），
以继续加载现有 GLM-5.3-Flash 权重，无需修改原始 `config.json`。

```python
from torchturbo.models.glm5_3_flash import (
    Glm5NextConfig,
    TurboGlm5NextForConditionalGeneration,
)
```

## 运行脚本

在项目根目录执行：

```bash
bash scripts/runs/npu_deps/glm5_3_flash_deps.sh
bash scripts/runs/lf_glm5_3_flash_text_sft_lora.sh
```

另一个训练入口为 `scripts/runs/lf_glm5_3_flash_text_sft_lora_8.sh`。
训练入口、参数和默认权重路径保持不变。
NPU KDA 环境和测试命令见 [NPU KDA 说明](../../glm5_3_flash_npu_kda.md)。

## 验证

```bash
PYTHONPATH=src python scripts/validate_glm5_3_flash.py --mode direct --device cpu
PYTHONPATH=src python -m pytest tests/models/glm5_3_flash -q
```

NPU 测试需先加载训练脚本中使用的 CANN 和自定义算子环境。
