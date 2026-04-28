#!/bin/bash

source /work/HHRI-AI/anaconda/etc/profile.d/conda.sh
conda activate saqlain_vllm

# 啟用 Microsoft Foundry 整合
export CLAUDE_CODE_USE_FOUNDRY=1

# Azure 資源名稱（將 {resource} 替換為您的資源名稱）
# export ANTHROPIC_FOUNDRY_RESOURCE={resource}
# 或提供完整的基礎 URL：
export ANTHROPIC_FOUNDRY_BASE_URL=https://yungh-mifup8jf-eastus2.openai.azure.com/anthropic
export ANTHROPIC_FOUNDRY_API_KEY=7eDjKNvsB9ry3GD11CoKkiIOpDFqNO4s5HWrFgUo184Ak5CYWsoKJQQJ99BKACHYHv6XJ3w3AAAAACOGfYEz
# 將模型設定為您資源的部署名稱
export ANTHROPIC_DEFAULT_SONNET_MODEL='claude-sonnet-4-6'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='claude-haiku-4-5'
export ANTHROPIC_DEFAULT_OPUS_MODEL='claude-opus-4-6'

     echo "Claude Code Chat"
     echo "================"
     echo "1) Resume last conversation"
     echo "2) New conversation"
     echo ""
     read -p "Choose [1/2]: " choice

     case $choice in
         1)
             claude --resume
             if [ $? -ne 0 ]; then
                 echo "No previous conversation found. Starting new conversation..."
                 claude
             fi
             ;;
         2)
             claude
             ;;
         *)
             echo "Invalid choice. Please enter 1 or 2."
             exit 1
             ;;
     esac
