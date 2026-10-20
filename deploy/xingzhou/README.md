# EL_Mask 最小生产部署

本目录只维护生产镜像和发布打包所需文件。测试代码保存在仓库的 `tests/`，
不进入镜像、Docker 构建上下文或部署 ZIP。本目录不含测试服务启动脚本、
密钥生成工具、旧签名实现或本地废仓。运行时会话不需要配置、挂载或保存密钥。

本次星舟部署使用平台已有的 **sam3_sdk** 镜像，不重新构建镜像。
Torch、Torchvision、Triton、CUDA/NVIDIA 依赖由该镜像提供；
应用依赖随 ZIP 离线交付。网页已确认该镜像可选，但未提供其内部版本信息，
不能把本地验证当作该镜像已通过验收。

## 生产内容边界

| 交付物 | 必须包含 | 明确排除 |
|---|---|---|
| 运行镜像 | Linux x86_64、Python 3.12 和 pip、已安装且可用的 Torch/Torchvision/CUDA、推理所需系统库及编译工具 | 不在启动时安装或升级 GPU 栈 |
| `/src` 部署 ZIP | `run.sh`、`requirements-offline.txt`、`wheels/`、`xingzhou_startup.py`、`xingzhou-config.json`、运行源码、自定义组件及编译资源、模型、词表、技能文档、LICENSE、文件摘要清单 | GPU wheels、测试、开发工具链、Docker 构建脚本、`.git`、`.runtime`、缓存、历史结果、密钥文件 |
| 模型 | `/src/models/sam3.pt`、`/src/models/big-lama.pt`；默认另行只读挂载 | 禁止复制进运行镜像；只有显式 `--with-models` 才写入 ZIP |
| 运行数据 | 可写 `/src/.runtime`、`/src/.gradio`、`/src/public_downloads`；缓存默认 `/src/.runtime/cache` | 不依赖宿主机用户的缓存或绝对个人路径 |

`runtime-files.txt` 是逐文件生产白名单。不要用 `COPY .` 或“把仓库全部打包”
替代它。SAM3 推理的现有导入链依赖 `sam3/train` 中少数数据结构、mask 和
matcher 模块，因此白名单保留这几个实际依赖文件；不会携带 trainer、训练
脚本、训练数据或整个训练目录。本次不通过改模型导入关系来缩减它们。
`app.py` 实际调用的 `scripts/layout_image_to_mask.py` 同样保留；其余 `scripts/`
文件均排除，目录名称本身不代表该文件是否属于生产依赖。
布局技能的 `SKILL.md`、`references/operation-catalog.md`、
`references/profile-priors.md` 和 `references/response-schema.json` 是
`load_skill_bundle()` 的运行输入，必须完整打包；不会放行其他 Markdown 文件。

应用现有的数据集评估功能仍需另行提供数据集；部署包不带宿主机数据集。
SAM3/LaMa 核心加载与推理使用 `/src/models`，不依赖个人模型路径。

## 构建与打包

在 `zhengqiyuan/xingzhou-deploy` 工作目录、Linux x86_64 的 Python 3.12 环境执行。
准备 wheels 时需要访问 PyPI；部署启动时不联网安装：

```sh
OUT="$(mktemp -d "$PWD/.runtime/production-dist.XXXXXX")"
python3 deploy/xingzhou/prepare_wheels.py "$OUT/wheels"
python3 deploy/xingzhou/bundle.py build "$OUT/sam3-el-mask.zip" \
    --ref HEAD --wheels-dir "$OUT/wheels" --with-models
python3 deploy/xingzhou/bundle.py verify "$OUT/sam3-el-mask.zip"
```

白名单和源码资源均从指定 Git **提交**读取，忽略工作目录里的未提交修改、
未跟踪文件及本地组件生成的类型桩。每个文件记录 SHA-256，ZIP 根目录直接
包含 `run.sh`，没有外层项目目录。发布必须先提交所需修改，再选择该提交打包。
如平台只能上传一个 ZIP，可加 `--with-models`：模型软链接会转换为真实文件
内容，模型来源以摘要为准；其余源码仍严格来自 Git 提交。

`requirements-runtime.txt` 保留直接应用依赖，根目录的 `requirements-offline.txt`
锁定完整应用依赖闭包。构建器要求每个依赖都有同版本 wheel，并拒绝 GPU 包、
缺包、多版本或未列入锁文件的包；wheel 的 SHA-256 同样进入 ZIP 清单。
`iopath` 在打包时从其纯 Python 源码包生成 wheel，运行时不编译源码。
`Dockerfile` / `build-image.sh` 是另行构建镜像的备用工具，本次不执行，
星舟也不会因为 ZIP 中有 Dockerfile 就自动构建它。

## 离线依赖

`run.sh` 首次启动执行：

```sh
python3 -m pip --isolated install --disable-pip-version-check \
    --no-index --find-links /src/wheels --only-binary=:all: --no-deps \
    --target /src/.runtime/python-deps-312-<lock-hash> --upgrade \
    -r /src/requirements-offline.txt
```

安装成功才写完成标记，再通过 `PYTHONPATH` 供主进程和模型子进程使用。
同一锁文件再次启动复用此目录；依赖版本变化或 Python 小版本变化使用新目录。
安装失败立即退出，不继续启动不完整的环境。不会修改镜像或宿主机的全局
site-packages，也不会递归下载 `timm` 所声明的 Torch/CUDA 依赖。

Web/CV 依赖并不全是纯 Python：NumPy、Pydantic Core、Pillow、OpenCV、
SciPy 等使用已编译 Linux wheel，不能换成 Windows wheel。本批包面向
CPython 3.12、Linux x86_64、glibc 2.28+；系统动态库、GPU 驱动、CUDA、
Torch/Torchvision 配套版本及 Triton 编译工具链仍由镜像/平台提供。
若 `sam3_sdk` 的 Python/架构不同，需要重新准备对应 wheel，不能靠离线
应用依赖替换解决。已验证的 GPU 组合为 Torch 2.7.0、Torchvision 0.22.0、
CUDA 12.6；这里描述的是 4090 验证环境，不是对平台镜像内容的断言。

## 平台启动约定

1. 将 ZIP 直接解压到 `/src`，选择平台默认的 `sam3_sdk` 镜像。
2. 提供两个模型文件；默认包使用只读挂载到 `/src/models` 的方式。
3. 运行用户应能写 `/src` 下的运行目录，包括离线依赖安装目标。
   实际 UID 由平台镜像决定，不能从备用 Dockerfile 推定挂载目录的权限。
4. 执行 `/bin/sh /src/run.sh`，固定监听 `0.0.0.0:8030`。
   这是平台约定的容器端口，不由 `SAM3_SERVER_PORT` 覆盖；外部访问端口由平台映射。
   入口依次查找 `python3`、`python`；本批 wheel 对应 Python 3.12。
5. 本包的 `xingzhou-config.json` 已配置星舟服务 **10259** 的实际访问地址：

```text
SAM3_SESSION_DEPLOYMENT_ID=sam3-xingzhou-10259
SAM3_SESSION_COOKIE_NAME=sam3_xingzhou_10259_owner
SAM3_ALLOWED_ORIGINS=http://10.120.105.205:30231
SAM3_SESSION_COOKIE_SECURE=0
SAM3_ALLOW_INSECURE_COOKIE=1
```

平台仅执行 `./run.sh` 时会加载此配置，不需要额外注入环境变量或密钥。
该地址是 2026-10-16 在“接口信息”核实的服务来源，不是管理页面
`10.120.105.243:30008`，也不包含 `/inferenceStart`。

写操作（POST/PUT/PATCH/DELETE）的来源校验不再依赖预先写好的地址：浏览器发来的
`Origin`（或 `Referer`）只要与该请求的 `Host` 或 `X-Forwarded-Host` 一致（同源）
即放行，因此平台重新分配 IP、端口或改用域名/HTTPS 时无需重新打包。
`SAM3_ALLOWED_ORIGINS` 改为可选的额外来源（逗号分隔，可留空），只在反向代理
改写了 `Host` 且未提供 `X-Forwarded-Host` 时才需要填写真实访问地址。
其他站点的页面无法伪造这两个请求头，跨站写请求仍返回 403。平台不能注入环境变量时，此 JSON
就是唯一配置入口：除上述五项外，也可写入其他字符串设置（例如
`LAYOUT_MASK_VLM_BASE_URL`），启动时按同样规则载入。
此 HTTP 配置只适用于当前内部验收，传输本身没有 HTTPS 加密。

环境变量优先于 JSON，显式空值或错误值仍会失败，不会被默认值掩盖。
会话检查日志同时列出各设置来自环境还是包内配置。
部署到其他服务时同时指定独立部署 ID、Cookie 名称、真实来源和协议配置：

```sh
export SAM3_SESSION_DEPLOYMENT_ID=el-production
export SAM3_SESSION_COOKIE_NAME=sam3_el_production_owner
export SAM3_ALLOWED_ORIGINS=https://actual-service.example
export SAM3_SESSION_COOKIE_SECURE=1
export SAM3_ALLOW_INSECURE_COOKIE=0
/bin/sh /src/run.sh
```

内部 HTTP 验收需显式设 `SAM3_SESSION_COOKIE_SECURE=0`、
`SAM3_ALLOW_INSECURE_COOKIE=1`。非 Secure Cookie 在 HTTPS 下同样可用；
`SAM3_SESSION_COOKIE_SECURE=1` 时额外来源不能是 `http://`。
不同服务使用不同 Cookie 名称；Cookie 本身不按端口隔离。
不配置任何会话签名密钥变量或挂载目录。

启动脚本只运行一个 Uvicorn worker，禁用自动重载。Cookie 凭证及内部临时校验材料只存在于进程内；
重启后旧页面须重新打开首页，不承诺恢复未保存的工作。

## 基础镜像环境报告

`run.sh` 在离线安装之前先执行 `xingzhou_envreport.py`，把平台镜像的原始环境写到
容器日志（标准错误），每段一行 JSON，`event` 为 `xingzhou_env`：

| `section` | 内容 |
|---|---|
| `system` | Python 版本与路径、OS、glibc、架构、运行 UID/GID、`/src` 是否可写及剩余空间、pip 版本 |
| `gpu` | `nvidia-smi` 的显卡、驱动、显存、算力，驱动支持的 CUDA 版本，`nvcc`，以及 `CUDA*`/`NV_*`/`NVIDIA*`/`LD_LIBRARY_PATH` 环境变量 |
| `torch` | 镜像自带的 Torch/Torchvision/Triton 版本、Torch 编译的 CUDA 与 cuDNN、可用架构、显卡算力及是否原生支持 BF16 |
| `packages` | 镜像预装的全部 Python 包版本，以及本包通过 `PYTHONPATH` 覆盖的同名包（镜像版本 vs 本包版本） |

报告只读、不联网，任何探测失败都只记为该字段的错误，不会阻止启动。环境变量只输出
上述名称前缀，名称含 KEY/TOKEN/SECRET/PASS/CRED/AUTH 的一律跳过。

## 启动自检与版本

入口先准备离线应用依赖，再做原有两项检查，通过后用同一个 Python 解释器执行 `sam3_demo.app`：

- 载入 `xingzhou-config.json` 并执行原有会话配置校验，`SAM3_SESSION_COOKIE_SECURE=1` 时拒绝 `http://` 额外来源。
- CUDA 是否可用、设备名称、可用显存及 BF16 能力；空闲显存少于 8192 MiB
  只警告，模型加载仍由原有 supervisor 决定，不在自检中加载模型或执行推理。

文件完整性由发布前的 `bundle.py verify` 负责（逐文件 SHA-256），启动时不再
比对清单，因此改配置后无需同步修改 `bundle-manifest.json`。缺依赖、缺模型或
运行目录不可写时，应用自身导入或初始化会直接报错退出。

`--check` 同样会在首次运行时离线安装应用依赖，但不会下载模型、
调用外部模型 API 或开启监听。可单独执行：

```sh
/bin/sh /src/run.sh --check
```

标准输出使用 `xingzhou_startup` JSON 事件记录检查阶段及结果。
失败写入标准错误并返回非零退出码，不打印环境全集、Cookie 值或密钥。
若存在 `bundle-manifest.json`，其 Git SHA 只作为 `SAM3_BUILD_GIT_SHA` 的默认值
写入日志版本标识，不做校验。

## 验收边界

平台探活应使用 `/healthz`，就绪检查使用 `/readyz`，并为模型加载设置启动宽限期。
不要把 `/` 或 `/inferenceStart` 作为探针路径；前者会创建浏览器会话，后者不是
本应用的健康检查接口。镜像中的 Docker `HEALTHCHECK` 不代表星舟已配置对应探针。

`/healthz` 检查服务存活，`/readyz` 同时报告模型状态；HTTP 200 不等于已完成
模型加载或推理。生产验收还需实际上传、SAM3 分割、LaMa 修复送入分割、
队列与导出操作，并用独立浏览器检查用户隔离和容器重启后会话失效。

宿主机需要已配置 Docker 执行权限和 NVIDIA Container Toolkit。此工具不改
Docker socket 权限、用户组、系统依赖，也不停止现有服务。当前 4090 用户
不能访问 Docker socket，因此可完成语法、打包和入口检查；实际镜像构建、
容器 GPU 推理及容器重启仍需管理员提供 Docker 执行权限或代执行。
静态检查通过不得写成容器验收通过。

## 依赖与验证

沿用 torch 2.7.0 / torchvision 0.22.0 / CUDA 12.6 的组合，不改宿主机环境。
容器使用 headless OpenCV；Pydantic 2.13.4 与开发环境 sam3 一致，满足 Gradio 6.14.0 的版本约束。
SAM3 模型构建器的导入链需要 `psutil`，模型下载辅助模块需要 `requests`，
两者显式列入运行依赖，不能假设宿主机安装会自动带进镜像。
应用传递依赖锁定在 `requirements-offline.txt`，wheel 摘要记录在 ZIP 清单。
平台基础镜像的版本和摘要尚待平台侧确认。测试应包含“缺应用依赖的隔离环境”
离线安装、应用启动和 GPU 栈来源核对，不能只检查已装好全部依赖的宿主环境。

以下命令只在开发仓库运行，测试本身不会被打包：

```sh
python3 tests/test_production_bundle.py
python3 tests/test_xingzhou_entrypoint.py
python3 tests/test_xingzhou_wheels.py
python3 -m unittest discover -s tests -p 'test_xingzhou_startup.py'
```
