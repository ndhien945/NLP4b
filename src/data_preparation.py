import os
import json
import random
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def aggregate_and_split(data_root="data", dataset_types=["UIT_HWDB_line", "UIT_HWDB_paragraph", "UIT_HWDB_word"], val_ratio=0.1, seed=42):
    """
    Hàm này duyệt qua các thư mục dữ liệu thô (gồm cả line và paragraph), 
    gom cụm nhãn từ các file label.json và phân chia thành các tập Train/Validation/Test.
    
    Tham số:
    - data_root: Thư mục gốc chứa dữ liệu (mặc định là 'data').
    - dataset_types: Danh sách các loại dữ liệu cần gom cụm.
    - val_ratio: Tỉ lệ chia cho tập Validation (mặc định 0.1 tức là 10%).
    - seed: Hạt giống ngẫu nhiên để đảm bảo việc chia tập luôn cố định ở các lần chạy khác nhau.
    """
    
    # Cố định random seed để kết quả chia tập không bị thay đổi mỗi lần chạy lại file này
    random.seed(seed)
    
    # Tạo thư mục chứa dữ liệu đã qua xử lý (nếu chưa tồn tại)
    processed_dir = os.path.join(data_root, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    all_train_samples = []
    test_samples = []

    # Duyệt qua từng loại dữ liệu (line và paragraph)
    for dtype in dataset_types:
        print(f"\n--- Đang xử lý tập: {dtype} ---")
        
        # -------------------------------------------------------------
        # 1. THU THẬP DỮ LIỆU HUẤN LUYỆN (TRAIN DATA)
        # -------------------------------------------------------------
        # Cấu trúc folder: data/UIT_HWDB_line/UIT_HWDB_line/train_data/
        train_raw_dir = os.path.join(data_root, dtype, dtype, "train_data")
        
        if os.path.exists(train_raw_dir):
            # Duyệt qua các thư mục con (ví dụ: 1, 2, ..., 250)
            for folder_id in os.listdir(train_raw_dir):
                folder_path = os.path.join(train_raw_dir, folder_id)
                
                # Bỏ qua nếu không phải là thư mục
                if not os.path.isdir(folder_path):
                    continue
                    
                label_path = os.path.join(folder_path, "label.json")
                if os.path.exists(label_path):
                    # Đọc file label.json trong từng thư mục con
                    with open(label_path, 'r', encoding='utf-8') as f:
                        labels = json.load(f)
                        
                    # Lặp qua từng cặp ảnh - nhãn trong file json
                    for img_name, text in labels.items():
                        # Tạo đường dẫn TƯƠNG ĐỐI tới file ảnh. 
                        # Việc dùng đường dẫn tương đối giúp code chạy mượt mà trên cả máy cá nhân lẫn Kaggle/Render
                        img_rel_path = os.path.join(dtype, dtype, "train_data", folder_id, img_name)
                        
                        all_train_samples.append({
                            "image": img_rel_path,
                            "text": text.strip() # Xóa khoảng trắng thừa ở 2 đầu văn bản
                        })
        else:
            print(f"[Cảnh báo] Không tìm thấy thư mục: {train_raw_dir}")

        # -------------------------------------------------------------
        # 2. THU THẬP DỮ LIỆU KIỂM THỬ (TEST DATA)
        # -------------------------------------------------------------
        test_raw_dir = os.path.join(data_root, dtype, dtype, "test_data")
        
        if os.path.exists(test_raw_dir):
            for folder_id in os.listdir(test_raw_dir):
                folder_path = os.path.join(test_raw_dir, folder_id)
                
                if not os.path.isdir(folder_path):
                    continue
                    
                label_path = os.path.join(folder_path, "label.json")
                if os.path.exists(label_path):
                    with open(label_path, 'r', encoding='utf-8') as f:
                        labels = json.load(f)
                        
                    for img_name, text in labels.items():
                        img_rel_path = os.path.join(dtype, dtype, "test_data", folder_id, img_name)
                        test_samples.append({
                            "image": img_rel_path,
                            "text": text.strip()
                        })
        else:
            print(f"[Cảnh báo] Không tìm thấy thư mục: {test_raw_dir}")

    # -------------------------------------------------------------
    # 3. XÁO TRỘN VÀ CHIA TẬP TRAIN / VALIDATION
    # -------------------------------------------------------------
    print("\n--- Tiến hành xáo trộn và chia tập dữ liệu ---")
    
    # Xáo trộn toàn bộ mẫu train để mô hình không học theo thứ tự ID thư mục
    random.shuffle(all_train_samples)
    
    # Tính toán số lượng mẫu dành cho tập Validation
    val_size = int(len(all_train_samples) * val_ratio)
    
    # Cắt mảng để chia tập
    val_samples = all_train_samples[:val_size]
    train_samples = all_train_samples[val_size:]
    
    print(f"Tổng số mẫu Train thu thập được: {len(all_train_samples)}")
    print(f"-> Đã chia thành: {len(train_samples)} mẫu Train và {len(val_samples)} mẫu Validation.")
    print(f"Tổng số mẫu Test: {len(test_samples)}")

    # -------------------------------------------------------------
    # 4. LƯU KẾT QUẢ RA FILE JSON
    # -------------------------------------------------------------
    train_out = os.path.join(processed_dir, "train_split.json")
    val_out = os.path.join(processed_dir, "val_split.json")
    test_out = os.path.join(processed_dir, "test_split.json")
    
    # ensure_ascii=False để lưu chuẩn tiếng Việt (không bị biến thành mã \u00e1)
    # indent=4 để format file json thụt lề dễ đọc
    with open(train_out, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=4)
    with open(val_out, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=4)
    with open(test_out, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=4)
        
    print(f"\n[Hoàn tất] Các file index đã được lưu thành công tại thư mục: {processed_dir}")

if __name__ == "__main__":
    # Điểm bắt đầu thực thi của script. 
    # Đảm bảo thư mục "data" nằm cùng cấp với nơi bạn chạy lệnh terminal
    aggregate_and_split(
        data_root="data", 
        dataset_types=["UIT_HWDB_line", "UIT_HWDB_paragraph", "UIT_HWDB_word"], 
        val_ratio=0.1
    )