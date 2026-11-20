# HTTP API 速查

服务默认监听 `127.0.0.1:8099`，交互式文档在 `/api/docs`。
所有写接口的 body 都是 JSON。

## 项目与状态

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/state` | 项目名/统计、类别表、settings（阈值）、class_colors、引擎与显存信息 |
| GET | `/api/stats` | 项目统计（按类别实例数、缺失文件数） |
| GET | `/api/projects` | 列出 `data/projects/*` 下所有项目 |
| POST | `/api/project/switch` | `{path}` 切换项目（服务后台重启，前端轮询后刷新） |
| POST | `/api/project/create` | `{name, source_dir, classes, do_copy, recursive, switch}` 从文件夹新建项目 |
| POST | `/api/classes` | `{classes:[...]}` 保存项目类别表 |
| POST | `/api/thresholds` | `{conf:{"car":0.6}}` 保存按类别置信度阈值 |
| POST | `/api/colors` | `{colors:{"car":"#ff3b30"}}` 保存按类别颜色 |
| POST | `/api/refilter` | `{apply:bool}` 按阈值重筛已有标注；`apply=false` 只返回影响统计 |
| POST | `/api/prune` | 清理原图已丢失的条目 |

## 图片与标注

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/images` | `?status=all\|pending\|done\|reviewed&q=&offset=&limit=`；返回 `total/offset/groups/items` |
| GET | `/api/images/{id}` | 单图标注（不含 RLE，掩码走 PNG 接口） |
| POST | `/api/images/{id}/segment` | `{classes, confidence, min_area, replace}` 按类别重跑 |
| POST | `/api/images/{id}/prompt` | `{text, confidence}` 追加一个提示词的结果 |
| POST | `/api/images/{id}/box_prompt` | `{box:[x0,y0,x1,y1], class_name, text_hint}` 框选补标 |
| POST | `/api/images/{id}/classes` | `{classes}` 写入该图的类别 |
| POST | `/api/images/{id}/status` | `{status: pending\|done\|reviewed}` |
| POST | `/api/images/{id}/instances/{iid}` | `{kept, class_name}` 隐藏/恢复、改类别 |
| DELETE | `/api/images/{id}/instances/{iid}` | 删除实例（服务端保留一份用于撤销） |
| POST | `/api/images/{id}/undelete` | 撤销最近一次删除 |
| GET | `/api/images/{id}/instances/{iid}/mask` | 单实例掩码 PNG（RGBA），参数 `color/alpha/max_size` |
| POST | `/api/images/{id}/instances/{iid}/mask` | `{w,h,add,sub}` 画笔精修（base64 PNG 涂抹层） |
| POST | `/api/images/{id}/clear` | 清空该图实例 |
| POST | `/api/images/remove` | `{ids, delete_files}` 批量移除（项目内副本才真删） |

## 媒体

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/media/image/{id}` | 原图（`?max_size=` 限制长边） |
| GET | `/media/thumb/{id}` | 缩略图（按源文件 mtime 自动失效） |
| GET | `/media/overlay/{id}` | 掩码叠加 PNG；`?hi=<实例id>` 高亮该实例并压暗其余；`?hide=&hcls=` 视图过滤 |
| GET | `/media/labelmap/{id}` | 实例 id 标签图（RGB 编码 id），前端按像素命中用 |

## 导入与导出

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/browse?dir=` | 目录浏览（当前层图片 + 视频 + 子目录） |
| POST | `/api/import` | `{paths, do_copy, dest, recursive}` 本机路径入库（文件夹默认只取该层） |
| POST | `/api/upload` | multipart 上传图片 |
| POST | `/api/export` | `{formats, only_reviewed, out, groups, images}` 导出 |
| GET | `/api/video/probe?path=` | 读视频宽高/fps/帧数/时长 |
| POST | `/api/video` | 视频导入 + 自动标注（返回 `job_id`） |

## 任务

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/jobs` | 当前所有任务 |
| GET | `/api/jobs/{id}` | 单个任务状态（state/progress/message/elapsed/results） |
| POST | `/api/jobs/{id}/pause` | 暂停 |
| POST | `/api/jobs/{id}/resume` | 继续 |
| POST | `/api/jobs/{id}/cancel` | 取消（保留已完成部分） |
| DELETE | `/api/jobs/{id}` | 从任务条移除（需先取消） |
| POST | `/api/annotate` | `{classes, confidence, overwrite, status, limit}` 批量标注（返回 `job_id`） |
