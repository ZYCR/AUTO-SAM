#!/usr/bin/env bash
# 下载 SAM3 权重与 BPE 词表
#
# 背景：官方权重在 HuggingFace(facebook/sam3) 是 gated（需申请 + token），
#      本脚本改用 ModelScope 的同名镜像（无需审批），并用官方仓库里的
#      merges.txt 还原 sam3 代码期望的 CLIP BPE 词表。
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p weights assets

CKPT=weights/sam3.pt
EXPECT=3450062241

if [ -f "$CKPT" ] && [ "$(stat -c%s "$CKPT")" -eq "$EXPECT" ]; then
  echo "权重已存在且大小正确：$CKPT"
else
  echo "下载 $CKPT (3.2GiB) ..."
  curl -L -C - --retry 5 --retry-delay 3 \
    -o "$CKPT" "https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt"
  size=$(stat -c%s "$CKPT")
  if [ "$size" -ne "$EXPECT" ]; then
    echo "大小不符：$size != $EXPECT，删掉重试" >&2
    exit 1
  fi
fi

BPE=assets/bpe_simple_vocab_16e6.txt.gz
if [ ! -f "$BPE" ]; then
  echo "下载 BPE 词表 ..."
  tmp=$(mktemp)
  curl -sL -o "$tmp" "https://modelscope.cn/models/facebook/sam3/resolve/master/merges.txt"
  # merges.txt 的内容就是 bpe_simple_vocab_16e6.txt，压成 gz 即可
  gzip -c "$tmp" > "$BPE"
  rm -f "$tmp"
fi
python3 -c "
import gzip, sys
n = len(gzip.open('$BPE','rt').read().split('\n'))
print('BPE 词表行数:', n)
assert n == 48896, 'BPE 词表行数不对（应为 48896）'
"
echo "完成：$CKPT ; $BPE"
