import os
from huggingface_hub import snapshot_download

def download_huggingface_model():
    # 模型仓库的 ID
    repo_id = "LIBERO-Safety/pi05_libero_safety"
    
    # 设置下载后保存在本地的文件夹路径
    local_dir = "/openbayes/input/input2/pi05_libero_safety"
    
    # 确保本地文件夹存在
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"开始从 Hugging Face 下载模型: {repo_id}")
    print(f"保存路径: {local_dir}")
    print("下载中，请稍候...")
    
    try:
        # 如果你启用了 hf_transfer 加速，可以通过环境变量强制开启
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        # 使用 snapshot_download 下载整个仓库
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # 设置为 False 以保存实际文件而非软链接
            # 如果该模型是私有模型或需要同意条款，取消下方注释并填入你的 HuggingFace Access Token
            # token="hf_xxxxxxxxxxxxxxxxxxxxxxxxx", 
            resume_download=True,         # 支持断点续传
            max_workers=8                 # 开启多线程下载加快速度
        )
        print("🎉 下载完成！")
        
    except Exception as e:
        print(f"❌ 下载过程中出现错误: {e}")

if __name__ == "__main__":
    download_huggingface_model()