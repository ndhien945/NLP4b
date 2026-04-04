import os
import sys
# --- BẮT ĐẦU ĐOẠN THÊM MỚI ---
# Chỉ đường cho Python biết thư mục gốc (project-root) nằm ở đâu
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Đẩy thư mục gốc vào danh sách tìm kiếm thư viện của Python
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
# --- KẾT THÚC ĐOẠN THÊM MỚI ---
import json
import torch
import math
import io
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from typing import Dict, List, Any, Tuple
from torch.nn.utils.rnn import pad_sequence

# Giả định bạn đã tải thư mục deepseek_ocr về máy/Kaggle bằng snapshot_download
from deepseek_ocr.modeling_deepseekocr import (
    format_messages,
    text_encode,
    BasicImageTransform,
    dynamic_preprocess,
)

class OCRJSONDataset(Dataset):
    """
    Module tải dữ liệu được tối ưu hóa để đọc từ file JSON index.
    """
    def __init__(self, json_path, data_root="data"):
        self.data_root = data_root
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Không tìm thấy file index: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
            
        print(f"Đã tải {len(self.samples)} mẫu từ {json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Nối data_root với đường dẫn tương đối trong file JSON để ra đường dẫn tuyệt đối
        img_path = os.path.join(self.data_root, sample["image"])
        text = sample["text"]
        
        # Mở ảnh và chuyển sang hệ màu RGB để đồng nhất
        try:
            img_obj = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Xử lý trường hợp ảnh lỗi để tránh sập toàn bộ epoch
            print(f"Lỗi đọc ảnh {img_path}: {e}")
            img_obj = Image.new('RGB', (100, 100), color='white') 
            text = "" # Bỏ qua mẫu này

        # Định dạng chuẩn hội thoại mà DeepSeek yêu cầu
        instruction = "<image>\nFree OCR. "
        conversation = [
            {
                "role": "<|User|>",
                "content": instruction,
                "images": [img_obj],
            },
            {
                "role": "<|Assistant|>",
                "content": text
            },
        ]
        return {"messages": conversation}


class DeepSeekOCRDataCollator:
    """
    Module gom nhóm (batching) và tiền xử lý ảnh tinh vi từ notebook.
    Bao gồm cắt ảnh động (crop_mode) và tính toán token padding.
    """
    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype  
        self.train_on_responses_only = train_on_responses_only

        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0

    # ... [Giữ nguyên toàn bộ logic các hàm deserialize_image, calculate_image_token_count, 
    # process_image, process_single_sample, và __call__ từ file notebook của bạn ở đây] ...
    # Để code ngắn gọn, tôi không in lại hàng trăm dòng code Collator hoàn hảo của bạn, 
    # bạn chỉ cần copy nguyên class DeepSeekOCRDataCollator từ notebook dán tiếp vào đây là chuẩn 100%.