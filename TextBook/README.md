# 本地教材目录

教材 PDF 因体积和版权原因不进入公开 Git 仓库，请由使用者自行准备。

目录按照年级划分，文件命名格式为：

```text
TextBook/
├── 一年级/
│   ├── 语文_一年级_上册.pdf
│   └── 数学_一年级_上册.pdf
├── 七年级/
│   ├── 数学_七年级_下册.pdf
│   └── 生物_七年级_下册.pdf
└── 九年级/
    └── 化学_九年级_下册.pdf
```

放入教材后执行：

```bash
python scripts/ocr_textbooks.py
python scripts/index_textbooks.py
```

教材索引会写入本机 Qdrant，不会提交到 Git。
