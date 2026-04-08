import os
import sys
import time

# Tắt cơ chế chặn lỗi thái quá của Unsloth
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

import torch
from transformers import AutoModel, Trainer, TrainingArguments
from unsloth import FastVisionModel, is_bf16_supported

# Định vị thư mục gốc để import các module local
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import các class nạp dữ liệu
from src.dataset import OCRJSONDataset, DeepSeekOCRDataCollator

def train_model(
    data_index_path="data/processed/train_split.json",
    image_root_path="/kaggle/input/datasets/danghiennguyen/datasetocr/data", # ĐIỀU HƯỚNG VÀO DATASET KAGGLE
    output_dir="models/deepseek_lora",
    lora_rank=16,
    learning_rate=2e-4,
    max_steps=60,
    batch_size=2
):
    print("=== KHỞI TẠO QUÁ TRÌNH HUẤN LUYỆN ===")
    
    print("1. Đang tải mô hình DeepSeek-OCR...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/DeepSeek-OCR",
        load_in_4bit=True,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=True,
        use_gradient_checkpointing="unsloth"
    )

    print(f"2. Áp dụng LoRA với rank r={lora_rank}...")
    model = FastVisionModel.get_peft_model(
        model,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
    )

    print(f"3. Đang nạp tập dữ liệu huấn luyện từ: {image_root_path}")
    # TÁCH BIỆT: Load file JSON từ thư mục làm việc, nhưng nạp ảnh từ thư mục Kaggle Input
    json_full_path = os.path.join(project_root, data_index_path)
    train_dataset = OCRJSONDataset(
        json_path=json_full_path, 
        data_root=image_root_path # <--- Điểm mấu chốt giải quyết lỗi
    )
    
    data_collator = DeepSeekOCRDataCollator(
        tokenizer=tokenizer,
        model=model,
        image_size=640,
        base_size=1024,
        crop_mode=True,
        train_on_responses_only=True,
    )

    FastVisionModel.for_training(model)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(project_root, output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=max_steps, 
        learning_rate=learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        args=training_args,
    )

    print("\n4. Bắt đầu huấn luyện...")
    start_time = time.time()
    
    trainer_stats = trainer.train()
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    print(f"\n[Hoàn tất] Thời gian huấn luyện: {training_time:.2f} phút.")

    save_path = os.path.join(project_root, output_dir)
    print(f"5. Đang lưu mô hình tại: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("=== HOÀN TẤT TOÀN BỘ QUÁ TRÌNH ===")

if __name__ == "__main__":
    train_model(max_steps=60, lora_rank=16)