# Doc-QA Assistant

基于本地知识库的智能问答系统，使用 RAG（检索增强生成）技术实现。通过向量检索从本地文档中召回相关内容，结合大语言模型生成带引用的回答。

## 项目结构
doc-qa-assistant/

├── data/

│   └── raw/

│       └── 东坡诗话.txt          # 原始文档（示例知识库）

├── chroma_db/                     # 向量数据库（自动生成，不提交）

├── scripts/

│   └── splitter.py                # 文本递归切分工具

├── pipeline.py                    # 核心流水线：入库、检索、问答

├── store.py                       # 文档入库脚本

├── app.py                         # Gradio 交互界面

├── eval_run.py                    # 评估脚本（20条测试问题）

├── run_comparison.py              # chunk_size 对比实验脚本

├── .env                           # API Key 配置（不提交）

└── README.md                      # 本文件
## 环境配置

### 依赖安装
bash

uv add python-dotenv openai chromadb langchain-text-splitters gradio
### 环境变量

在项目根目录创建 `.env` 文件，填入 API Key：
SILICONFLOW_API_KEY=your_siliconflow_api_key_here

## 使用方法

### 1. 文档入库

将待入库的文档放入 `data/raw/` 目录，然后运行：
bash

uv run python pipeline.py

默认入库 `data/raw/东坡诗话.txt`，支持自定义文件路径。

### 2. 启动 Web 界面
bash

uv run python app.py

访问地址：`http://localhost:7860`

### 3. 运行评估
bash

uv run python eval_run.py

评估包含 20 条测试问题（15 条文档内 + 5 条文档外），自动统计正确率。

## 切分参数对比实验

| chunk_size | chunk_overlap | 文档内正确数 (15条) | 文档外拒答数 (5条) | 总正确率 |
|------------|---------------|---------------------|---------------------|----------|
| 256        | 80            | 7/15                | 5/5                 | 60%      |
| 512        | 80            | 7/15                | 5/5                 | 60%      |
| 1024       | 80            | 8/15                | 5/5                 | 65%      |

> **实验条件**：使用 `RecursiveCharacterTextSplitter` 递归切分，嵌入模型 `BAAI/bge-m3`，LLM 为 `deepseek-ai/DeepSeek-V3`，温度 0.3，最大 token 2048。

### 实验结论

- 三档 chunk_size 均实现了**文档外问题 100% 拒答**，模型严格遵守了知识库边界。
- 文档内正确率随 chunk_size 增大略有提升（60% → 65%），1024 表现最佳。
- 当前瓶颈在于检索召回精度，后续可通过优化切分策略或调整检索参数进一步提升。

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 向量数据库 | ChromaDB |
| 嵌入模型 | BAAI/bge-m3（硅基流动 API） |
| 大语言模型 | DeepSeek-V3（硅基流动 API） |
| 文本切分 | RecursiveCharacterTextSplitter |
| Web 界面 | Gradio 6.0 |
| 编程语言 | Python 3.11 |

## 踩坑记录

### Python 版本
- Python 3.14 存在兼容性问题，需降级至 3.11。

### 依赖安装
- `onnxruntime` 需要安装 VC++ 运行库。
- `NumPy` 需降级至 1.26.4 以避免兼容性警告。

### ChromaDB
- 更换嵌入模型后，需删除 `chroma_db` 目录重新入库。

### Gradio 6.0
- `respond` 函数返回值需使用字典格式：`{"role": "user"/"assistant", "content": ...}`。
- `gr.Chatbot` 不再支持 `type="messages"` 参数。
- `theme` 参数需移至 `launch()` 方法中。
- `server_port` 参数值不能加引号。

### 路径问题
- 使用 `pathlib.Path` 管理路径，避免 Windows 反斜杠兼容性问题。
- 运行脚本时需确保当前工作目录为项目根目录。

## License

MIT