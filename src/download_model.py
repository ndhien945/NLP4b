import os
import sys
from huggingface_hub import snapshot_download
    


if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    
def download_deepseek_code():
    # Đảm bảo thư mục lưu trữ nằm ở thư mục gốc của dự án (project-root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    target_dir = os.path.join(project_root, "deepseek_ocr")
    
    print(f"Đang tải mã nguồn mô hình về: {target_dir}...")
    
    # Tải các file cần thiết (mã nguồn và cấu hình) từ HuggingFace
    snapshot_download(
        repo_id="unsloth/DeepSeek-OCR", 
        local_dir=target_dir,
        # Chỉ tải các file code .py và file config, bỏ qua các file model .safetensors nặng
        # vì chúng ta chỉ cần code architecture ở bước load dataset này
        allow_patterns=["*.py", "*.json"] 
    )
    
    # Tạo một file __init__.py trống để Python nhận diện thư mục này là một package
    init_file = os.path.join(target_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            pass
            
    print("Hoàn tất! Cấu trúc mã nguồn mô hình đã sẵn sàng.")

if __name__ == "__main__":
    download_deepseek_code()