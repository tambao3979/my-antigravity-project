# 🔥 AI Camera Fire Detection System

Phần mềm camera AI sử dụng YOLOv8 để phát hiện đám cháy và vật thể theo thời gian thực. Khi lửa được phát hiện liên tục trong **5 giây**, hệ thống gửi cảnh báo kèm ảnh qua **Telegram**.

---

## 📁 Cấu trúc dự án

```
rogue-perseverance/
├── gui.py                # Entry point GUI
├── config.py             # Cấu hình trung tâm
├── detector.py           # YOLO model wrapper
├── fire_tracker.py       # Logic xác nhận cháy 5 giây
├── telegram_notifier.py  # Gửi thông báo Telegram
├── camera.py             # Vòng lặp camera chính
├── requirements.txt      # Thư viện phụ thuộc
├── .env.example          # Template cấu hình
└── .env                  # Cấu hình của bạn (tạo thủ công)
```

---

## ⚙️ Cài đặt

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Tạo file `.env`

```bash
copy .env.example .env
```

Mở file `.env` và điền thông tin:

```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_CHAT_ID=987654321
AUTH_ADMIN_PASSWORD=mat_khau_admin_manh
AUTH_USER_PASSWORD=mat_khau_user_manh
```

> **Cách lấy Bot Token:** Nhắn tin cho [@BotFather](https://t.me/BotFather) → `/newbot`
>
> **Cách lấy Chat ID:** Nhắn tin cho [@userinfobot](https://t.me/userinfobot) hoặc dùng `/getUpdates`

### 3. Kiểm tra tên nhãn "fire" trong model

Nếu model của bạn dùng tên nhãn khác (ví dụ: `flame`, `Fire`), hãy sửa trong `.env`:

```env
FIRE_CLASS_NAME=fire
```

---

### 🚀 Chạy giao diện cửa sổ (Khuyên dùng)

Hệ thống cung cấp một giao diện (GUI) đẹp mắt để quản lý và cấu hình dễ dàng.
```bash
python gui.py
```

## 🎮 Điều khiển

| Phím | Hành động |
|------|-----------|
| `Q` hoặc `ESC` | Thoát chương trình |

---

## 📋 Thông số mặc định

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `FIRE_CONFIRM_SECONDS` | `5` | Giây liên tục cần thiết để xác nhận cháy |
| `ALERT_COOLDOWN_SECONDS` | `60` | Khoảng cách tối thiểu giữa 2 thông báo |
| `CONFIDENCE_THRESHOLD` | `0.5` | Ngưỡng độ tin cậy YOLO |
| `CAMERA_INDEX` | `0` | Index camera (0 = webcam tích hợp) |

---

## 🖥️ Giao diện camera

- 🟥 **Khung đỏ** – Phát hiện lửa
- 🟩 **Khung xanh** – Vật thể khác
- 🟠 **Khung cam** – Đang đếm ngược xác nhận
- 📊 **Thanh tiến trình** (phía dưới) – Đếm ngược 5 giây
- 🚨 **Banner đỏ** – Đám cháy đã được xác nhận

---

## 🔔 Thông báo Telegram

Khi xác nhận cháy, bot sẽ gửi:
- 📸 Ảnh chụp từ camera tại thời điểm phát hiện
- 📅 Thời gian phát hiện
- 🔍 Số vùng lửa được phát hiện
- ⚠️ Cảnh báo khẩn cấp

---

## 🛠️ Cấu hình nâng cao

Chỉnh sửa file `.env` để thay đổi:

```env
# Camera ngoài USB
CAMERA_INDEX=1

# Tăng độ nhạy (giảm ngưỡng)
CONFIDENCE_THRESHOLD=0.4

# Xác nhận nhanh hơn (3 giây)
FIRE_CONFIRM_SECONDS=3
```
