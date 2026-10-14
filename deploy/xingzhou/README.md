# EL_Mask 最小生产部署

本目录只维护生产镜像和发布打包所需文件。测试代码保存在仓库的 `tests/`，
不进入镜像、Docker 构建上下文或部署 ZIP。本目录不含测试服务启动脚本、
密钥生成工具、旧签名实现或本地废仓。运行时会话不需要配置、挂载或保存密钥。

## 生产内容边界

| 交付物 | 必须包含 | 明确排除 |
|---|---|---|
| 运行镜像 | Python 3.12、CUDA 对应的 PyTorch wheels、固定的 Web/推理运行依赖、8030 入口及健康检查 | 项目源码、模型、测试/验收工具、Git、个人目录、会话密钥 |
| `/src` 部署 ZIP | `run.sh`、`sam3_demo` 运行代码、SAM3 必需模块、运行辅助模块、自定义组件后端与已编译资源、BPE 词表、LICENSE、文件摘要清单 | `tests`、训练入口、评测工具、notebooks、前端开发工具链、Docker 构建脚本、文档、`.git`、`.runtime`、缓存、历史结果、`.orig`、废仓、密钥文件 |
| 模型 | `/src/models/sam3.pt`、`/src/models/big-lama.pt`；默认另行只读挂载 | 禁止复制进运行镜像；只有显式 `--with-models` 才写入 ZIP |
| 运行数据 | 可写 `/src/.runtime`、`/src/.gradio`、`/src/public_downloads`；缓存默认 `/src/.runtime/cache` | 不依赖宿主机用户的缓存或绝对个人路径 |

`runtime-files.txt` 是逐文件生产白名单。不要用 `COPY .` 或“把仓库全部打包”
替代它。SAM3 推理的现有导入链依赖 `sam3/train` 中少数数据结构、mask 和
matcher 模块，因此白名单保留这几个实际依赖文件；不会携带 trainer、训练
脚本、训练数据或整个训练目录。本次不通过改模型导入关系来缩减它们。

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
5. 配置真实浏览器访问来源和 Cookie 安全属性：

```sh
SAM3_SESSION_DEPLOYMENT_ID=el-production
SAM3_SESSION_COOKIE_NAME=sam3_el_production_owner
SAM3_ALLOWED_ORIGINS=https://actual-service.example
SAM3_SESSION_COOKIE_SECURE=1
```

内部 HTTP 验收需显式设 `SAM3_SESSION_COOKIE_SECURE=0`、
`SAM3_ALLOW_INSECURE_COOKIE=1`，并把允许来源改成实际端口地址。
不同服务使用不同 Cookie 名称；Cookie 本身不按端口隔离。
不配置任何会话签名密钥变量或挂载目录。

启动脚本会先检查模型和 BPE 文件，再检查上述配置。只运行一个 Uvicorn
worker，禁用自动重载。Cookie 凭证及内部临时校验材料只存在于进程内；
重启后旧页面须重新打开首页，不承诺恢复未保存的工作。

## 验收边界

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
删除应用对 Cookie 签名包的直接依赖。基础镜像与全部传递依赖尚未按摘要锁定，
构建产物会在 `/opt/sam3/installed-requirements.txt` 记录实际安装版本。

以下命令只在开发仓库运行，测试本身不会被打包：

```sh
python3 tests/test_production_bundle.py
python3 tests/test_xingzhou_entrypoint.py
```
