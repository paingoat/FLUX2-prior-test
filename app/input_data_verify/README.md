# Kiểm tra input Prior test

Viewer đọc `data/Prior_test/<thao tác>/<case>/metadata.json` và hiện từng mẫu: prompt, input, ground truth.

```bash
python app/input_data_verify/server.py
python app/input_data_verify/server.py --port 8502
```

Mở http://127.0.0.1:8501

Từ laptop qua SSH:

```bash
ssh -L 8501:localhost:8501 user@host
```

Bấm **Tải lại** để đọc lại metadata và file ảnh. Thumbnail JPEG nằm trong `app/input_data_verify/.cache/` (cần Pillow; nếu không có Pillow thì route thumbnail trả ảnh gốc).
