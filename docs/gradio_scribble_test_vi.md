# Test scribble trên Gradio với prompt ngắn

Mục tiêu: dùng nét vẽ để xác định **vùng/đối tượng**, text chỉ mô tả **hành động và nội dung mới**. Đây là thử khả năng có sẵn của FLUX.2 Klein, chưa huấn luyện riêng cho scribble.

## 1. Chạy trên MMLab

```bash
cd /home/nvtiep/project/mix-modality/FLUX2-prior-test
git fetch origin
git switch experiment/klein-multi-reference-scribble
git pull --ff-only
conda activate flux2-klein
CUDA_VISIBLE_DEVICES=0 python app/gradio_inference/app.py \
  --preload distill --port 7860 --no-share
```

Nếu thiếu VRAM, thêm `--offload`. Nếu chưa cài môi trường, xem phần MMLab trong README chính; cần đặt `HF_HOME` trong `.env` đúng thư mục cache trên máy lab.

Trên laptop, mở terminal khác (thay `HOST_MMLAB` bằng địa chỉ SSH thực tế):

```bash
ssh -L 7860:localhost:7860 nvtiep@HOST_MMLAB
```

Giữ kết nối SSH và mở http://localhost:7860 trong trình duyệt. Các ảnh nằm trên server: tải ảnh sạch từ các đường dẫn bên dưới về laptop trước khi upload vào Gradio. Đường dẫn ảnh trong tài liệu tính từ thư mục repo; đây là ảnh `images/`, không phải `ground_truth/`.

## 2. Quy trình cho mỗi ảnh

1. Upload ảnh sạch vào **Image and scribble**.
2. Vẽ đường bao mảnh quanh đối tượng cần sửa. Với thao tác thêm, khoanh vùng trống nơi đối tượng mới sẽ xuất hiện. Không tô kín ảnh.
3. Nhập **Name** theo ví dụ, **Prompt** đúng một câu bên dưới.
4. Chọn **Multi-reference (original + overlay)**, model **distill**, seed **42**. Distill dùng 4 steps, guidance 1.0 cố định.
5. Xem **Prompt sent to model**, bấm **Run** rồi chờ kết quả và file ZIP xuất hiện.
6. Giữ nguyên ảnh, nét vẽ, prompt, seed và model; đổi sang **Overlay only**, đổi hậu tố Name từ `_multi` sang `_overlay`, chạy lại.

Chế độ multi-reference tự thêm phần hướng dẫn vai trò ảnh 1/ảnh 2, xóa nét vẽ và bảo toàn vùng ngoài. **Prompt nhập ngắn không có nghĩa toàn bộ prompt gửi model chỉ có một câu.** Overlay only giữ nguyên prompt người dùng. Vì vậy, phép so sánh này đánh giá cả chiến lược ảnh và prompt, chưa cô lập riêng tác động của số ảnh.

Nếu muốn thử base, giữ cùng nét vẽ, chọn **base**, dùng mặc định 50 steps/guidance 4.0, đổi Name có thêm `_base`. Base sẽ tải khi được chọn chạy lần đầu; có thể mất thời gian. Chạy từng model hoặc dùng `--offload` nếu thiếu VRAM.

## 3. Mười ảnh và prompt để copy

Các prompt dưới đây được rút gọn từ mục tiêu edit đã trao đổi; chúng không tái hiện mọi chi tiết trong prompt dài của metadata gốc. Bắt đầu với nét đỏ, trừ mẫu áo đỏ dùng nét xanh lá.

### 01 — Add / Background insertion

- Ảnh: `data/Prior_test/add/background_insertion/images/input_28821.png`
- Name: `28821_add_background_distill_multi`
- Vẽ: oval đỏ nằm ngang trên mặt nước ở nửa trái ảnh, dưới đường bờ hồ; tránh nhánh cây nổi gần đáy ảnh. Oval thể hiện vị trí và kích thước tương đối của thuyền.

```text
Add one dark green wooden canoe in the marked region.
```

Kiểm tra: đúng một thuyền, nằm trong vùng đánh dấu; cảnh hồ giữ nguyên và không còn nét vẽ.

### 02 — Add / Object cloning

- Ảnh: `data/Prior_test/add/object_cloning/images/input_271920.png`
- Name: `271920_add_clone_distill_multi`
- Vẽ: oval đỏ nằm ngang trên khoảng trắng phía trên, lệch về phía gót giày bên trái. Đây là vị trí chiếc giày mới; không khoanh chiếc gốc.

```text
Add a second shoe matching the existing shoe in the marked region.
```

Kiểm tra: có hai chiếc cùng kiểu dáng; chiếc ban đầu còn nguyên. Vùng trống ảnh khá hẹp nên chú ý kích thước và việc che khuất tự nhiên.

### 03 — Add / Spatial placement

- Ảnh: `data/Prior_test/add/spatial_placement/images/input_43028.png`
- Name: `43028_add_person_distill_multi`
- Vẽ: oval đỏ đứng trên bờ tuyết bên trái cột gỗ, giữa cột và tường bờ kè. Không khoanh cột hoặc lòng kênh.

```text
Add one person in a dark winter coat walking away in the marked region.
```

Kiểm tra: người đứng đúng bờ tuyết; cột, cầu và cây không bị thay đổi.

### 04 — Change / Color change

- Ảnh: `data/Prior_test/change/color_change/images/input_kf3etr.png`
- Name: `kf3etr_change_color_distill_multi`
- Vẽ: dùng **xanh lá**, khoanh phần áo polo đỏ của bé gái. Tránh khoanh cả khuôn mặt và cánh tay; không tô kín áo.

```text
Change the marked shirt to royal blue.
```

Kiểm tra: chỉ áo đổi màu; giữ nếp vải, da, mặt và nền.

### 05 — Change / Local style transfer

- Ảnh: `data/Prior_test/change/local_style_transfer/images/input_284489.png`
- Name: `284489_change_style_distill_multi`
- Vẽ: khoanh đỏ riêng người trên phao trong màn hình iPod nằm ngang bên phải. Dùng nét nhỏ vì đối tượng rất bé; không khoanh cả iPod.

```text
Turn the marked person into a stylized 3D animated character.
```

Kiểm tra: chỉ nhân vật chuyển phong cách; phao, sông và vỏ iPod giữ nguyên. Đây là mẫu khó do kích thước đối tượng nhỏ.

### 06 — Change / Material change

- Ảnh: `data/Prior_test/change/material_change/images/input_167464.png`
- Name: `167464_change_material_distill_multi`
- Vẽ: đường đỏ bao quanh mặt tiền ốp gỗ, không mở rộng sang cây hoặc bầu trời. Cửa sổ/lan can nằm trên mặt tiền vẫn cần giữ nguyên.

```text
Change the marked wooden siding to matte dark charcoal metal cladding.
```

Kiểm tra: vật liệu thay đổi nhưng cửa sổ, lan can, hình học tòa nhà và bóng cây được giữ.

### 07 — Remove / Single-object scene

- Ảnh: `data/Prior_test/remove/single_object_scene/images/input_39818.png`
- Name: `39818_remove_single_distill_multi`
- Vẽ: khoanh đỏ quanh toàn bộ quả đỏ trên mặt phẳng sáng, chừa một ít nền quanh quả.

```text
Remove the marked object.
```

Kiểm tra: quả và bóng liên quan được xóa; nền khớp xung quanh, mép cong của mặt phẳng không biến dạng.

### 08 — Remove / Multi-object scene

- Ảnh: `data/Prior_test/remove/multi_object_scene/images/input_kf1ihk.png`
- Name: `kf1ihk_remove_multiobject_distill_multi`
- Vẽ: khoanh đỏ người mặc quân phục xanh, đội mũ tối màu bên trái cạnh cửa máy bay, từ đầu xuống chân. Tránh nhóm người bắt tay ở giữa.

```text
Remove the marked person.
```

Kiểm tra: xóa đúng người, tái tạo thân máy bay/mặt đất và giữ các người còn lại. Có thể đổi người được khoanh nhưng giữ nguyên prompt để kiểm tra khả năng chọn mục tiêu bằng nét vẽ.

### 09 — Replace / Single-object scene

- Ảnh: `data/Prior_test/replace/single_object_scene/images/input_149747.png`
- Name: `149747_replace_single_distill_multi`
- Vẽ: khoanh đỏ toàn bộ thằn lằn trên thân cây, gồm cả đuôi dài kéo sang trái.

```text
Replace the marked animal with a small green tree frog.
```

Kiểm tra: có ếch thay thế, không còn đuôi/chân thằn lằn; giữ nền và thân cây.

### 10 — Replace / Multi-object scene

- Ảnh: `data/Prior_test/replace/multi_object_scene/images/input_196584.png`
- Name: `196584_replace_multiobject_distill_multi`
- Vẽ: bao đỏ mép tấm biển lớn “WE ARE The 99%”, tránh người cầm biển và đám đông.

```text
Replace the marked sign with an LED display reading "CLIMATE ACTION NOW".
```

Kiểm tra: thay đúng biển, giữ đám đông; đánh giá lỗi sinh chữ riêng với lỗi chọn vùng.

Prior_test trong repo hiện không có nhóm relocate. Không lấy ảnh VIBE đã có nét đánh dấu làm ảnh sạch cho phép thử hai ảnh này.

## 4. Tự động lưu kết quả

Mỗi lần **Run thành công**, app tự lưu; không cần bấm Save:

```text
output/gradio/<name>_<timestamp>_<unique-id>/
  original.png
  overlay.png
  result.png
  visual.png
  config.json
  metadata.jsonl
output/gradio_exports/<name>_<timestamp>_<unique-id>.zip
```

- `original.png`: ảnh nền sạch; `overlay.png`: ảnh đã gộp nét vẽ.
- `result.png`: ảnh model sinh, không phải ground truth.
- `visual.png`: ba cột ảnh gốc → overlay → kết quả.
- `config.json`: Name, prompt nhập, prompt thực tế, mode, thứ tự ref và thông số chạy.
- `metadata.jsonl`: một dòng cho mỗi run, dùng các cột `original_file_name`, `overlay_file_name`, `result_file_name`, `visual_file_name`, `reference_file_names` theo định dạng ImageFolder nhiều ảnh.
- ZIP chứa sáu file của run, tải qua ô **Download run (images + metadata ZIP)**. Có thể giữ nguyên ZIP hoặc giải nén thành thư mục trước khi upload.

Tên thư mục có ID riêng để không ghi đè khi chạy lại cùng tên. Run lỗi inference sẽ không tạo bản kết quả thành công. Dữ liệu output được gitignore, không đi theo `git push` code.

## 5. Đẩy lên Hugging Face khi đã kiểm tra

App chỉ lưu local; thao tác upload do bạn thực hiện. Tạo một **Dataset repository** trên Hugging Face trước, thay `YOUR_USERNAME/YOUR_DATASET` bằng repo của bạn.

Từ terminal MMLab đã kích hoạt môi trường, đăng nhập rồi upload các thư mục run mới tạo (không upload `.env` hoặc toàn bộ repo code):

```bash
hf auth login
python - <<'PY'
from huggingface_hub import HfApi

HfApi().upload_folder(
    repo_id="YOUR_USERNAME/YOUR_DATASET",
    repo_type="dataset",
    folder_path="output/gradio",
    path_in_repo="runs",
    allow_patterns=["*/original.png", "*/overlay.png", "*/result.png",
                    "*/visual.png", "*/config.json", "*/metadata.jsonl"],
)
PY
```

Chỉ các run mới có `metadata.jsonl` được chuẩn bị theo định dạng này. Nếu thư mục chứa cả run cũ chưa có metadata, hãy chọn/copy các run mới sang một thư mục riêng rồi đổi `folder_path` sang thư mục đó. Có thể upload lại để bổ sung run mới. Nếu muốn dùng ZIP, upload các ZIP đã chọn trong `output/gradio_exports` thay cho thư mục run; tránh upload cả hai bản của cùng run vào cùng bộ dữ liệu.

Kiểm tra local bằng thư viện `datasets` (cài riêng nếu môi trường chưa có):

```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="output/gradio", split="train")
print(ds.column_names)
print(len(ds))
```

Nguồn định dạng: https://huggingface.co/docs/datasets/main/image_dataset — hỗ trợ nhiều cột ảnh với hậu tố `_file_name` và danh sách ảnh `_file_names`. Chưa kiểm chứng upload/viewer trên tài khoản Hugging Face trong lần sửa code này.

## 6. Ghi nhận đánh giá

Với mỗi cặp `_multi` / `_overlay`, ghi nhận: đúng vùng, đúng action, xóa hết nét đánh dấu, giữ vùng ngoài và chất lượng hòa trộn. Xem thời gian/VRAM trong `config.json`; dùng cùng cấu hình preload/offload khi so sánh. Mười mẫu × hai mode = 20 run cho một model, hoặc 40 run nếu chạy cả distill và base. Kết quả được lưu là đầu ra thử nghiệm, chưa tự động được chọn làm dữ liệu train.
