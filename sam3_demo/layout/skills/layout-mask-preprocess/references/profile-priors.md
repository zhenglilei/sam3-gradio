# Layout Profile Priors

这些规则是文字 few-shot 和经验起点，不是固定答案。必须结合当前图像和用户要求判断。

## ACT

常见特征：外轮廓存在大量细小凹坑或局部坑洼，但主体线宽基本正确，不同器件仍需保持分离。

推荐基线：

```text
threshold=12
invert=false
open_kernel=0
close_kernel=15
morph_pixels=0
min_component_area=0
region_mode=all
```

尺度参考：

- 很小缺口：`close=5/7`
- 轻度凹坑：`close=11/13`
- 常见 ACT 小凹坑：`close=15`
- 用户要求“再填一点”：`close=17`
- 较深凹坑：`close=19`
- `close>=21`：要求人工检查器件粘连
- `close>=25`：高风险，默认不要推荐

规则：填补小凹坑优先修改 close 并保持 morph=0；“再填一点”在当前有效 close 上增加约 2；“减少粘连”减少 2-4；出现真实孔洞消失或相邻器件连接时降低 close 并设置人工复核。

文字映射：

```text
边缘少量浅凹坑 -> close=11或13, morph=0
边缘大量典型小凹坑 -> close=15, morph=0
当前close=15且要求再填一点 -> close=17, morph=0
当前close=19且器件粘连 -> close=15或17, manual_review=true
```

## GE1

常见特征：线条整体偏细，主体轮廓通常连续，结构间隙需要保留，不应默认继承 ACT 的大 close。

推荐基线：

```text
threshold=12
invert=false
open_kernel=0
close_kernel=0
morph_pixels=1
min_component_area=0
region_mode=all
```

规则：轻微偏细用 `morph=+1`；明显偏细用 `morph=+2` 并人工复核；只修短断口且不要加粗时用 `close=3/5, morph=0`；用户说不要变粗时禁止正 morph；孔洞缩小或器件连接时降低 morph。

文字映射：

```text
线条整体轻微偏细 -> morph=+1, close=0
线条整体明显偏细 -> morph=+2, close=0, manual_review=true
少量短断口但不要整体变粗 -> close=3或5, morph=0
```

## GE2

常见特征：方形孔洞、内部空白和器件分隔间隙很重要，轮廓通常不需要大幅填补，对 close 和正 morph 敏感。

推荐基线：

```text
threshold=12
invert=false
open_kernel=0
close_kernel=0
morph_pixels=0
min_component_area=0
region_mode=all
```

规则：优先保留孔洞和间隙；不默认使用 ACT 的 close=15 或正 morph；用户要求保留孔洞时保持 close=0、morph<=0；色彩提取不完整时只在当前值附近小幅调整 threshold，每轮优先调整 2；无法判断缺失结构是孔洞还是噪声时保持当前参数并人工复核。

文字映射：

```text
方孔和间隔清楚 -> close=0, morph=0
明确要求保留方孔 -> close=0, morph=0
色彩较浅导致部分轮廓缺失 -> threshold在当前值附近调整2，其他参数不变
```

## Unknown

无法可靠区分 profile、图像同时存在冲突问题、图像质量不足或结构与已有 profile 不同时使用 Unknown。

规则：以当前参数为中心，每轮优先只改变一个参数；不自动套用 ACT close=15 或 GE1 morph=+1；不确定时保持当前参数、降低 confidence 并设置 `manual_review=true`。

文字映射：

```text
类型不明确但有轻微独立噪点 -> open=3，其他参数不变，manual_review=true
类型和问题均不明确 -> 参数保持不变，降低confidence，manual_review=true
```
