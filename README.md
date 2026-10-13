# paper-reader

一个面向团队研究资料库的轻量阅读器，统一管理论文、arXiv、OpenReview、项目网站、技术博客和新闻文章，并支持浏览器内阅读、Prompt 分析、协作与后台任务处理。

## 主要功能

- 统一资料库：自动扫描 `PDF / .doc / .docx`，并将网页正文保存为带元数据的 Markdown 快照
- 左侧紧凑论文列表，右侧大阅读区
- PDF 在线阅读、`.docx` 文本预览、`.doc` 原文件打开
- Prompt 管理：可配置多个 Prompt，结果落盘为本地 Markdown
- 后台任务队列：自动处理新论文，也可批量补跑历史论文
- 离线阅读包：导出原论文 + Markdown + 本地 HTML 阅读器
- 个人已读流程：每个用户都可以把论文标记为“已读”，只影响自己的列表视图
- Sources 页面：浏览外部抓取的论文归档，支持按天打包下载或一键导入主阅读器
- Insights 页面：基于已有“核心解读”生成历史脉络，并用时间线方式可视化
- 登录保护：支持本地用户名密码，也支持可选的飞书 OAuth 登录

## 配置

运行时配置从项目根目录的 `.env` 读取，进程环境变量优先级更高；也可用 `PAPER_READER_ENV_FILE` 指向另一个 dotenv 文件。`.env` 已被忽略，不能提交真实凭据。

### 服务与存储

默认值均相对于**启动命令的当前工作目录**：

- `PAPER_READER_HOST=0.0.0.0`
- `PAPER_READER_PORT=8000`
- `PAPER_READER_DATA_DIR=./data`
- `PAPER_READER_LIBRARY_DIR=./data/library`
- `PAPER_READER_SOURCES_ARCHIVE_DIR=./data/sources/huggingface_daily`
- `PAPER_READER_MAX_UPLOAD_BYTES=104857600`（默认 100 MiB）

`PAPER_READER_LIBRARY_DIR` 和 `PAPER_READER_SOURCES_ARCHIVE_DIR` 覆盖各自目录；未设置时从 `PAPER_READER_DATA_DIR` 派生。主论文库、SQLite 协作数据、Prompt 结果和 Sources 归档都在这些目录中，不随 Git 提交。

### 身份与管理员初始化

没有内置用户名、密码或 Flask session secret。生产环境必须设置稳定的 `PAPER_READER_SECRET_KEY`。未设置时应用为当前进程生成随机 session secret 并记录警告；重启后所有 session 会失效。

可选的本地管理员登录必须同时设置以下两项：

```dotenv
PAPER_READER_LOGIN_USERNAME=admin
PAPER_READER_LOGIN_PASSWORD=replace-with-a-unique-password
```

若未配置这对凭据，应用不在启动时创建本地账号；第一个成功注册或首次飞书登录的用户会成为管理员。也可显式指定管理员用户名列表：

```dotenv
PAPER_READER_ADMIN_USERS=admin,operations-admin
```

该列表仅是部署配置，取代了源代码中的命名管理员。连续输错本地密码 3 次会锁定 5 分钟。

### 飞书 OAuth

启用飞书登录需要：

```dotenv
PAPER_READER_SECRET_KEY=replace-with-a-random-secret
PAPER_READER_BASE_URL=https://reader.example.com
PAPER_READER_FEISHU_APP_ID=cli_xxx
PAPER_READER_FEISHU_APP_SECRET=xxx
# 可选；默认 https://open.feishu.cn
PAPER_READER_FEISHU_API_BASE_URL=https://open.feishu.cn
```

回调地址固定为 `https://<你的域名>/auth/feishu/callback`，并须在飞书应用的安全设置中登记。当前应用仍保留本地用户名密码登录与注册；飞书应用的可访问范围由飞书侧配置控制。

### LLM（DexGEM LiteLLM 网关）

所有论文解读、标签生成、聊天和 Insights 调用都经由团队的 OpenAI 兼容 LiteLLM 网关；服务不依赖本机 CLI、登录态或子进程。生产环境需要配置专用 virtual key：

```dotenv
PAPER_READER_LLM_BASE_URL=https://litellm.dex-gem.ai/v1
PAPER_READER_LLM_API_KEY=replace-with-paper-reader-virtual-key
# 长上下文主模型：论文解读、聊天、Insights
PAPER_READER_LLM_MODEL=glm-5.3
# 低成本首轮模型：自动标签
PAPER_READER_LLM_CHEAP_MODEL=deepseek-flash
# 允许 SQLite 中保存的 Prompt 模型覆盖默认值的名单
PAPER_READER_LLM_ALLOWED_MODELS=claude-sonnet-5-5,claude-sonnet-5,kimi-k3,glm-5.3,deepseek-flash
# 单进程请求启动限速、单次 HTTP 超时和每篇文档传给模型的保守字符/Token 上限
PAPER_READER_LLM_QPS=10
PAPER_READER_LLM_TIMEOUT=120
PAPER_READER_LLM_CONTEXT_TOKENS=12000
```

默认 base URL、模型、限速和超时如上。未知的 SQLite Prompt 模型会记录警告并回退到 `PAPER_READER_LLM_MODEL`。HTTP 聊天使用网关 SSE 增量响应，并为每次浏览器流式会话发送 `x-litellm-session-id`。文档请求先抽取 PDF/DOCX/Markdown 文本；有 arXiv Markdown 缓存时优先使用缓存；超过上下文预算时保留开头和结尾，省略中间正文。

### 批量导入飞书历史链接

`scripts/import_links.py` 读取每行一个对象的 JSONL：

```json
{"url":"https://example.org/post","chat_name":"DexGEM Research","msg_time_iso":"2026-01-02T03:04:05Z","context_text":"推荐阅读","sender_open_id":"ou_xxx"}
```

先做无副作用的分类检查，再执行导入：

```bash
.venv/bin/python scripts/import_links.py links.jsonl --dry-run
.venv/bin/python scripts/import_links.py links.jsonl --as-user admin
```

脚本直接使用与 Web UI 相同的 action queue、资料库和 SQLite store。默认 resume 行为会跳过已有 canonical URL；加 `--no-resume` 可刷新现有网页快照。每条飞书出处以 `source_type=feishu` 保存 chat name、消息上下文、发送者 open ID 和时间。

## 环境准备

建议使用 Python 3.11+。

### 1) 创建虚拟环境

```bash
python3 -m venv .venv
```

### 2) 安装依赖

```bash
.venv/bin/pip install -r requirements.txt
```

## 如何启动

### 方式一：直接启动

```bash
.venv/bin/python run.py
```

部署前请在项目根目录创建 `.env`，至少设置稳定的 session secret；如需本地密码登录，再同时设置用户名和密码：

```bash
cat > .env <<'EOF'
PAPER_READER_SECRET_KEY=replace-with-a-long-random-value
PAPER_READER_LOGIN_USERNAME=admin
PAPER_READER_LOGIN_PASSWORD=replace-with-a-unique-password
EOF
```

启动后访问：

- 本机：`http://127.0.0.1:8000`
- 局域网其他机器：`http://<你的机器IP>:8000`

### 方式二：在 tmux 中启动

如果你希望服务在后台持续运行，推荐用 tmux：

```bash
tmux new-session -d -s paper-reader 'cd /path/to/paper-reader && .venv/bin/python run.py'
```

查看状态：

```bash
tmux attach -t paper-reader
```

停止服务：

```bash
tmux kill-session -t paper-reader
```

## 项目结构

```text
paper-reader/
├── run.py
├── paper-reader-source/        # 外部来源抓取子项目
├── requirements.txt
├── src/paper_reader/
│   ├── app.py
│   ├── task_queue.py
│   ├── ai_summary.py
│   ├── prompt_manager.py
│   ├── offline_package.py
│   ├── source_archive.py
│   ├── static/
│   └── templates/
└── tests/

./data/                         # 默认数据根目录（相对于启动时 cwd）
├── library/                    # 主论文库、SQLite 和生成结果
└── sources/huggingface_daily/  # 外部来源归档
```

## 论文库说明

支持格式：

- `.pdf`
- `.doc`
- `.docx`

说明：

- `.pdf`：浏览器内联阅读
- `.docx`：会提取文本并做浏览器预览
- `.doc`：通常不直接内联渲染，但仍可索引并打开原文件

## Prompt 批量补跑

“历史论文批量补跑”面板默认只显示 **你还没标记为已读的论文**。

如果你希望把已完成论文也纳入批量列表，可以在该面板中勾选：

- `把我已读的论文也加入这次批量生成`

这样就会把你自己已经标记为已读、但又想重新处理的论文一起显示出来。

## Sources 页面

系统现在支持一个独立的 `Sources` 页面，用来浏览外部归档的论文来源数据。

当前已接入：

- Hugging Face Daily Papers

在主阅读器底部工具区可以进入 `Sources` 页面。这个页面支持：

- 按年 / 月 / 日浏览抓取归档
- 查看每天保存下来的论文列表
- 打包下载某一天选中的 PDF
- 把选中的论文直接导入主阅读器

导入后的目标目录默认是：

- `Sources/HuggingFace/YYYY/MM/DD/`

如果阅读器里启用了自动 Prompt，那么从 `Sources` 导入的论文也会自动进入后台任务队列。

### Sources 数据目录

默认从 `./data/sources/huggingface_daily/`（相对于启动时 cwd）读取外部归档；用 `PAPER_READER_SOURCES_ARCHIVE_DIR` 可以覆盖它。该目录是运行数据，不会提交到 Git。

### Hugging Face 抓取子项目

仓库里包含一个单独的子项目：

- `paper-reader-source/`

它负责：

- 抓取 Hugging Face Daily Papers
- 按天保存 manifest
- 下载对应 PDF 到本地归档目录

直接运行一次采集：

```bash
PYTHONPATH=.:paper-reader-source .venv/bin/python -m paper_reader_source.service \
  --run-on-start --once
```

长期运行定时服务：

```bash
PYTHONPATH=.:paper-reader-source .venv/bin/python -m paper_reader_source.service
```

或者直接用仓库里自带的脚本：

```bash
paper-reader-source/scripts/run_huggingface_daily_service.sh
```

当前调度规则：

- 时区：`Asia/Shanghai`
- 每天执行时间：`18:30`
- 默认保留 `upvotes >= 5` 的论文

## Insights 页面

系统现在支持一个独立的 `Insights` 页面，当前先聚焦 AutoResearch 的第一步：

- 历史上的研究脉络
- 哪些论文是奠基点 / 转折点 / 主流化节点
- 这个方向为什么出现，以及为什么后来会切换路线
- 最近 30 / 60 / 90 天里哪些 topic / method / benchmark 正在升温
- 接下来哪些方向值得做：未解问题、薄弱 claim、拥挤方向、高回报低密度方向、跨领域迁移空白

这个页面会：

- 读取你已经生成好的 `核心解读`
- 从已有论文库里筛出主题主线
- 在后台生成一版历史脉络数据，并持续把进度写回磁盘
- 在后台生成一版 `Momentum Radar`，梳理最近 30 / 60 / 90 天的研究热区
- 在后台生成一版 `Opportunity Map`，产出更接近 next-bet memo 的研究机会判断
- 用时间线 / 阶段卡片 / 年度热度条做可视化
- 用 trend radar / momentum dashboard 的形式展示最近的增长主题、被 follow 的方法路线、benchmark 焦点和边缘信号
- 用 opportunity map / research gap report 的形式展示哪些方向值得继续下注
- 在页面上显示当前进度，支持 `停止` / `继续`
- 即使浏览器关闭、刷新，或者 Web 服务重启，重新进入后也能看到最新状态并继续任务

当前入口：

- 主阅读器顶部导航
- 主阅读器底部工具区
- `Sources` 页面顶部

## 手动扫描 Paper 文件夹

如果你经常直接在文件系统里操作论文目录，比如：

- 手动复制 PDF 进去
- 在系统文件管理器里重命名文件
- 删除某些论文

可以在页面的“文件管理”区域点击：

- `快速扫描 Paper 文件夹`

这个功能的设计目标是：

- 只做文件扫描和目录更新
- 不自动运行任何 Prompt
- 不调用任何 AI 模型
- 不做重型、耗时的解析

也就是说，它更适合在你手动改动了论文目录之后，快速让网页里的论文列表同步更新。

## 离线阅读包

可以导出一个 zip 包，里面包含：

- 原始 PDF / Word 文件
- 已生成的 Markdown 结果
- 一个可离线打开的 `index.html`

解压后，直接双击 `index.html` 即可离线阅读。

## 测试

运行测试：

```bash
.venv/bin/python -m unittest tests.test_app
```

## 开发备注

- 默认运行数据写到启动目录下的 `data/`，不会跟随仓库提交
- 日志目录 `logs/` 默认不会提交到 Git
- 运行过程中生成的本地状态文件也已加入 `.gitignore`
