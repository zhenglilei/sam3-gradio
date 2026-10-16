# EL_Mask 最小生产部署

本目录只维护生产镜像和发布打包所需文件。测试代码保存在仓库的 `tests/`，
不进入镜像、Docker 构建上下文或部署 ZIP。本目录不含测试服务启动脚本、
密钥生成工具、旧签名实现或本地废仓。运行时会话不需要配置、挂载或保存密钥。

## 生产内容边界

| 交付物 | 必须包含 | 明确排除 |
|---|---|---|
| 运行镜像 | Python 3.12、CUDA 对应的 PyTorch wheels、固定的 Web/推理运行依赖、8030 入口及健康检查 | 项目源码、模型、测试/验收工具、Git、个人目录、会话密钥 |
| `/src` 部署 ZIP | `run.sh`、`xingzhou_startup.py`、`xingzhou-config.json`、`sam3_demo` 运行代码、SAM3 必需模块、运行辅助模块、自定义组件后端与已编译资源、BPE 词表、布局运行技能文档、LICENSE、文件摘要清单 | `tests`、训练入口、评测工具、notebooks、前端开发工具链、Docker 构建脚本、非运行文档、`.git`、`.runtime`、缓存、历史结果、`.orig`、废仓、密钥文件 |
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

在 EL_Mask 工作目录执行（Python 打包器仅用标准库）：

```sh
OUT="$(mktemp -d "$PWD/.runtime/production-dist.XXXXXX")"
python3 deploy/xingzhou/bundle.py build "$OUT/sam3-el-mask.zip" --ref HEAD
python3 deploy/xingzhou/bundle.py verify "$OUT/sam3-el-mask.zip"
sh deploy/xingzhou/build-image.sh "$OUT/sam3-runtime.tar"
```

白名单和源码资源均从指定 Git **提交**读取，忽略工作目录里的未提交修改、
未跟踪文件及本地组件生成的类型桩。每个文件记录 SHA-256，ZIP 根目录直接
包含 `run.sh`，没有外层项目目录。发布必须先提交所需修改，再选择该提交打包。
如平台只能上传一个 ZIP，可加 `--with-models`：模型软链接会转换为真实文件
内容，模型来源以摘要为准；其余源码仍严格来自 Git 提交。

`.dockerignore` 只放行 `Dockerfile` 和 `requirements-runtime.txt`。
`build-image.sh` 使用本目录作构建上下文，产出仅含运行环境的镜像。
可用 `SAM3_RUNTIME_IMAGE` 设置镜像标签。已有 ZIP/TAR 不会被覆盖。

## 平台启动约定

1. 将 ZIP 直接解压到 `/src`，导入或选用构建好的运行镜像。
2. 提供两个模型文件；默认包使用只读挂载到 `/src/models` 的方式。
3. 运行用户应能写 `/src` 下的运行目录。镜像默认 UID/GID 为 `10001:10001`；
   绑定宿主机目录时需由部署方预先匹配目录所有权，或选择有相应写权限的 UID。
4. 执行 `/bin/sh /src/run.sh`，固定监听 `0.0.0.0:8030`。
   这是平台约定的容器端口，不由 `SAM3_SERVER_PORT` 覆盖；外部访问端口由平台映射。
   入口优先使用显式 `PYTHON_BIN`，未指定时依次查找 `python3`、`python`。
   显式解释器不可执行时直接失败，不自动切换环境；解释器仍须满足 Python 3.12+。
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
`10.120.105.243:30008`，也不包含 `/inferenceStart`。平台若重新分配地址，
必须更新此 JSON 后提交并重新打包，或通过环境变量覆盖；不会自动信任任意来源。
此 HTTP 配置只适用于当前内部验收，传输本身没有 HTTPS 加密。

环境变量优先于 JSON，显式空值或错误值仍会失败，不会被默认值掩盖。
旧配置中的 `REPLACE_WITH_SERVICE_HOST` 等占位值会在启动时拒绝；删除残留的
环境变量可恢复使用包内配置。会话检查日志同时列出各设置来自环境还是包内配置。
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
`SAM3_ALLOW_INSECURE_COOKIE=1`，并把允许来源改成实际端口地址。
不同服务使用不同 Cookie 名称；Cookie 本身不按端口隔离。
不配置任何会话签名密钥变量或挂载目录。

启动脚本只运行一个 Uvicorn worker，禁用自动重载。Cookie 凭证及内部临时校验材料只存在于进程内；
重启后旧页面须重新打开首页，不承诺恢复未保存的工作。

## 启动自检与版本

入口先检查以下内容，通过后用同一个 Python 解释器执行 `sam3_demo.app`：

- 包清单、必需文件及大小，包括模型、词表、技能文档和五个组件的编译入口。
- 缓存、临时文件、导出、反馈、布局、日志等运行目录的实际写入权限。
  此项在重型依赖导入前执行；配置模块初始化目录失败时也会报告具体路径。
  入口不会更改挂载目录归属、放宽权限或绕过只读挂载，需部署方提供可写目录。
- Python 3.12+、核心依赖、SAM3 模型构建器和本包组件的实际导入。
- 原有会话配置校验，并要求来源协议与 Secure Cookie 配置一致。
- CUDA 是否可用、设备名称、可用显存及 BF16 能力；空闲显存少于 8192 MiB
  只警告，模型加载仍由原有 supervisor 决定，不在自检中加载模型或执行推理。

自检不会安装依赖、下载模型、调用外部模型 API 或开启监听。可单独执行：

```sh
/bin/sh /src/run.sh --check
```

标准输出使用 `xingzhou_startup` JSON 事件记录检查阶段、实际依赖版本及结果。
失败写入标准错误并返回非零退出码，不打印环境全集、Cookie 值或密钥。
Git SHA 从 `bundle-manifest.json` 读取并设置 `SAM3_BUILD_GIT_SHA`，因此无 Git 的
镜像中操作日志也能识别版本；如果外部注入了冲突 SHA，启动会拒绝。
此入口用于解压后的发布包，直接在源码树运行时缺少清单会失败，应先打包解压。
启动自检只核对文件存在性和大小；完整 SHA-256 检查由发布前的 `bundle.py verify`
执行，避免每次启动重复读取数 GB 权重。没有改变公共会话模块、算法或 UI。

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
容器使用 headless OpenCV；Pydantic 2.12.5 满足 Gradio 6.14.0 的版本约束。
SAM3 模型构建器的导入链需要 `psutil`，模型下载辅助模块需要 `requests`，
两者显式列入运行依赖，不能假设宿主机安装会自动带进镜像。
删除应用对 Cookie 签名包的直接依赖。基础镜像与全部传递依赖尚未按摘要锁定，
构建产物会在 `/opt/sam3/installed-requirements.txt` 记录实际安装版本。

以下命令只在开发仓库运行，测试本身不会被打包：

```sh
python3 tests/test_production_bundle.py
python3 tests/test_xingzhou_entrypoint.py
python3 -m unittest discover -s tests -p 'test_xingzhou_startup.py'
```
