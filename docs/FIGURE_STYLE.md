# 全文图件字体标准

图1、图2和后续所有主图、补充图统一采用 **Arial**。范围包括标题、面板编号、坐标轴、刻度、图例、表格以及化学结构原子文字。各图已确认的布局、字号层级和数据保持原样；统一字体不等同于全部文字使用同一个字号。

共享配置为 `configs/figure_style.json`，Python入口为 `scripts/figure_style.py`。新绘图脚本在设置其他rcParams后调用 `apply_font()`；RDKit绘图使用 `font_path()` 返回的字体文件。不要静默替换字体。字体文件不随仓库分发；绘图主机必须已安装Arial。当前Windows环境已验证Arial和Arial Bold，CI的制图步骤在Windows运行，Linux继续执行全部数值测试与建模数据复现。

```python
from figure_style import FONT_FAMILY, apply_font, font_path
apply_font()
# RDKit:
# drawer.drawOptions().fontFile = str(font_path())
```

图1和图2的SVG文字保留为Arial文字，PDF嵌入对应字体。化学结构还提供独立SVG；其原子文字的字体族同样明确为Arial。
