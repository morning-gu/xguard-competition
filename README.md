# XGuard 护栏揭榜赛 - 项目基础结构

## 赛事简介

**XGuard护栏揭榜赛**是【大模型安全撬壳计划】系列赛事之一，由阿里巴巴集团安全部主办。赛事致力于为广大AI安全爱好者与开发者提供一个开放协作的创意舞台，通过持续的模型测评、衍生二创与安全加固，推动"XGuard护栏"进入越用越强、越用越广的良性发展轨道。

## 赛事奖励

- **总奖金池**: 10万元现金 + 5万元礼品
- **赛道一（逆袭吧！创二代！）**: TOP 10 选手瓜分 50,000 元现金大奖
- **赛道二（破圈吧！抓码者！）**: 50,000元二创奖 + 丰富礼品

## 两大赛道

### 赛道一：逆袭吧！创二代！

**适合人群**: 算法高手、模型训练专家、想挑战技术高度的极客

**难度系数**: ⭐⭐⭐⭐⭐

**玩法规则**: 
- 基于 XGuard 开源数据集（允许扩充新数据）
- 结合 XGuard 模型或其他开源模型（10B 及以下）
- 训练并提交专属的安全护栏模型

### 赛道二：破圈吧！抓码者！

**适合人群**: 技术博主、创意开发者、自媒体达人、对AI安全感兴趣均可玩

**难度系数**: ⭐⭐（上手容易，获奖率高）

**支线A: 我是布道师**
- 玩法：撰写深度评测报告，并发布在各大平台

**支线B: 代码也疯狂**
- 玩法：基于 XGuard 进行应用开发或创意二创

## 参赛收益

- 💰 真金白银：50,000元二创奖 + 50,000元破圈奖
- 🎁 周周有惊喜：MAC MINI、千问眼镜、MacBook Neo 等总价值 5 万的装备
- 💻 算力自由：参赛即送 20 小时 GPU 算力，TOP 20 必得 100 小时
- 🚀 官方证书：TOP 20 专属阿里巴巴认证荣誉证书
- 📝 学生专属：简历诊断、招聘直通机会

## 官方资源

- **开源数据集**: [XGuard-Train-Open-200K](https://modelscope.cn/datasets/Alibaba-AAIG/XGuard-Train-Open-200K)
- **开源模型**: 
  - [YuFeng-XGuard-Reason-8B](https://modelscope.cn/models/Alibaba-AAIG/YuFeng-XGuard-Reason-8B)
  - [YuFeng-XGuard-Reason-0.6B](https://modelscope.cn/models/Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B)
- **赛事说明书**: [查看完整文档](https://alidocs.dingtalk.com/i/nodes/MNDoBb60VLYDGNPytPzQr5wnJlemrZQ3)

## 联系方式

- **钉钉群号**: 168975002855 (AAIG开源交流群)
- **微信公众号**: 添加开源小助手 + 发送"参赛"，邀您进微信群

## 主办单位

- 阿里安全
- AAIG (Alibaba AI Guard)

## 项目结构说明

```
xguard-competition/
├── README.md              # 项目说明文档
├── docs/                  # 文档目录
│   ├── competition_info.md    # 赛事详细信息
│   ├── submission_guide.md    # 提交指南
│   └── evaluation_metrics.md  # 评估指标说明
├── src/                   # 源代码目录
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型定义
│   ├── training/          # 训练脚本
│   └── inference/         # 推理脚本
├── data/                  # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── external/          # 外部数据
├── models/                # 模型目录
│   ├── pretrained/        # 预训练模型
│   ├── checkpoints/       # 训练检查点
│   └── submissions/       # 提交模型
├── scripts/               # 工具脚本
├── notebooks/             # Jupyter Notebooks
├── tests/                 # 测试代码
├── requirements.txt       # Python 依赖
└── .gitignore            # Git 忽略文件
```

## 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd xguard-competition
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 下载模型 (可选)

**方式一：使用 ModelScope SDK**
```bash
python scripts/download_model.py
```

**方式二：推理时自动下载**
推理脚本会自动从 ModelScope 下载模型，无需手动下载。

### 4. 运行推理示例
```bash
python src/inference/infer_yufeng_xguard.py
```

### 5. 查看使用文档
详细使用说明请查看 [模型使用指南](docs/model_usage.md)

## 重要链接

- [ModelScope 魔搭社区](https://modelscope.cn/)
- [赛事页面](https://modelscope.cn/events/197/赛事介绍)
- [大模型安全撬壳计划](https://s.alibaba.com/aichallenge)

---

**注意**: 本项目为参加 XGuard 护栏揭榜赛而创建的基础框架，参赛者可根据具体需求进行调整和扩展。
