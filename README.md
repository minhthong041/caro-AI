# Caro AI

Caro AI là một game Caro/Gomoku 15x15 viết bằng Python và Pygame, có tích hợp các tác nhân AI để người chơi có thể đấu với máy, thu thập dữ liệu ván chơi và thử nghiệm mô hình học máy dự đoán nước đi.

Dự án kết hợp hai hướng tiếp cận:

- **Minimax + Alpha-Beta pruning** cho AI tìm kiếm truyền thống.
- **CNN supervised learning** để học nước đi từ dữ liệu ván chơi.

## Tính năng

- Giao diện game bằng Pygame.
- Chơi người với người để thu thập dữ liệu.
- Chơi người với AI Minimax.
- Chơi người với CNN đã huấn luyện.
- Lưu lịch sử ván chơi thành file JSON.
- Huấn luyện CNN từ log ván chơi.
- Tăng dữ liệu train bằng các phép xoay/lật bàn cờ.
- Lưu lịch sử train CNN thành JSON/CSV và biểu đồ nếu có matplotlib.
- Benchmark tự động các AI để đo win rate, số nước và thời gian suy nghĩ.
- Unit test cho luật chơi, Minimax và dữ liệu huấn luyện.

## Công nghệ sử dụng

- Python 3.11
- Pygame
- NumPy
- TensorFlow
- Keras
- unittest

Các dependency được khai báo trong [requirements.txt](requirements.txt).

## Cấu trúc dự án

```txt
caro_ai_project/
├── ai/
│   ├── minimax.py        # AI Minimax và Alpha-Beta pruning
│   ├── supervised.py     # CNN, xử lý dữ liệu train, dự đoán nước đi
│   ├── evaluate.py       # Benchmark các tác nhân AI không cần mở Pygame
│   └── train_cnn.py      # Script huấn luyện CNN
├── assets/
│   ├── fonts/            # Font giao diện
│   └── images/           # Ảnh nền, quân cờ, nút bấm
├── data/
│   ├── human_logs/       # Log ván chơi mới
│   ├── logs_archive/     # Log đã dùng để train
│   ├── models/           # Model CNN sau khi train
│   └── training_history/ # JSON/CSV/PNG lịch sử train CNN
├── src/
│   ├── board.py          # Logic bàn cờ và kiểm tra thắng
│   ├── config.py         # Cấu hình bàn cờ
│   ├── logger.py         # Ghi log ván chơi
│   ├── main.py           # GameController và vòng lặp chính
│   └── ui.py             # Giao diện Pygame
├── tests/
│   ├── __init__.py
│   └── test_logic.py     # Unit test
├── requirements.txt
└── README.md
```

## Chế độ chơi

| Chế độ | Mô tả |
|---|---|
| VS MINIMAX | Người chơi đấu với AI Minimax. |
| VS TRAINED CNN | Người chơi đấu với CNN đã huấn luyện. |
| PvP TRAINING | Hai người chơi với nhau để tạo dữ liệu train. |

Chế độ CNN cần model tại:

```txt
data/models/caro_supervised.h5
```

Nếu chưa có file model tại đường dẫn trên, hãy tạo log ván chơi trước rồi chạy script huấn luyện. Game vẫn mở được khi thiếu TensorFlow/Keras hoặc thiếu model, nhưng chế độ CNN sẽ báo lỗi cho tới khi model được tạo.

## Luật chơi

- Bàn cờ có kích thước 15x15.
- Ô trống được biểu diễn bằng `0`.
- Người chơi 1 được biểu diễn bằng `1`.
- Người chơi 2 được biểu diễn bằng `2`.
- Người chơi thắng khi có từ 5 quân liên tiếp trở lên theo hàng ngang, dọc hoặc chéo.

Luật thắng được cấu hình trong [src/config.py](src/config.py):

```python
WIN_LENGTH = 5
```

## Cài đặt

Clone repository:

```powershell
git clone <repository-url>
cd caro_ai_project
```

Tạo môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài dependency:

```powershell
pip install -r requirements.txt
```

Dự án đã được kiểm thử với Python 3.11.

## Chạy game

Từ thư mục gốc của dự án:

```powershell
python src\main.py
```

Sau đó chọn chế độ chơi trong menu chính.

## Minimax AI

AI Minimax nằm trong [ai/minimax.py](ai/minimax.py).

Minimax giả lập các nước đi tiếp theo của hai bên:

- AI chọn nước đi có điểm cao nhất.
- Đối thủ được giả định sẽ chọn nước đi gây bất lợi nhất cho AI.
- Alpha-Beta pruning loại bỏ các nhánh không cần xét để giảm thời gian tính toán.
- Iterative deepening thử lần lượt từ depth 1 tới depth tối đa.
- Time budget giúp AI trả về nước tốt nhất đã tìm xong nếu hết thời gian.
- Zobrist Hashing dùng khóa 64-bit cập nhật bằng XOR cho transposition table.
- Quiescence search mở rộng thêm các vị trí còn nước thắng, chặn hoặc threat mạnh khi depth chính đã hết.

Để phù hợp với bàn cờ 15x15, AI chỉ sinh các nước đi ứng viên gần những quân đã xuất hiện trên bàn. Khi bàn cờ trống, AI ưu tiên đánh vào trung tâm.

Hàm đánh giá bàn cờ dựa trên:

- Số quân liên tiếp.
- Số đầu mở của một chuỗi quân.
- Các thế như 2, 3, 4 quân liên tiếp.
- Các pattern nâng cao như open four, blocked four, open three, broken three.
- Double threat, tức một nước tạo từ hai mối đe dọa trở lên.
- Trọng số phòng thủ để ưu tiên chặn các chuỗi nguy hiểm của đối thủ.

Pattern heuristic dùng nhóm mẫu cố định và cache theo chuỗi dòng đã mã hóa để giảm chi phí tạo lại pattern trong vòng lặp đánh giá.

## CNN AI

CNN nằm trong [ai/supervised.py](ai/supervised.py).

Mô hình nhận trạng thái bàn cờ và trả về xác suất cho 225 vị trí có thể đánh. Khi chọn nước đi, chương trình chỉ xét các ô còn trống để tránh nước đi không hợp lệ.

Model mới dùng kiến trúc Residual CNN: stem Conv2D, các residual block có skip-connection, policy head 1x1 và Dense softmax. Kiến trúc này sâu hơn CNN tuần tự ban đầu nhưng vẫn giữ output theo từng ô trên bàn.

Dữ liệu train được chuẩn hóa theo góc nhìn của người chơi hiện tại:

- Quân của người chơi hiện tại thành `1`.
- Quân đối thủ thành `2`.
- Ô trống giữ nguyên `0`.

Với model mới, bàn cờ sau chuẩn hóa được mã hóa thành 3 kênh nhị phân:

- Kênh 1: quân của người chơi hiện tại.
- Kênh 2: quân đối thủ.
- Kênh 3: ô trống.

Cách này giúp CNN nhìn từng loại ô như một đặc trưng riêng, thay vì xem mã quân `2` như một giá trị số lớn hơn mã quân `1`. Code vẫn tương thích với model `.h5` cũ nhận input 15x15 scalar.

Mỗi mẫu dữ liệu được mở rộng thành 8 biến thể bằng các phép xoay và lật bàn cờ. Cách này giúp mô hình học tốt hơn từ cùng một lượng log ban đầu.

Pipeline train chỉ dùng các nước đi hợp lệ từ nguồn phù hợp như `human` hoặc `minimax`, đồng thời bỏ qua dữ liệu lỗi, nước đi vào ô đã có quân hoặc nước tự sinh trực tiếp từ CNN.

## Thu thập dữ liệu

Chọn chế độ:

```txt
PvP TRAINING
```

Sau mỗi ván, log sẽ được lưu vào:

```txt
data/human_logs/
```

Mỗi file log gồm kết quả ván, chế độ chơi, tổng số nước đi và danh sách trạng thái bàn cờ trước từng nước.

## Huấn luyện CNN

Sau khi có dữ liệu trong `data/human_logs/`, chạy:

```powershell
python ai\train_cnn.py
```

Nếu muốn tạo lại model Residual CNN mới với encoding 3 kênh thay vì tiếp tục model `.h5` cũ:

```powershell
python ai\train_cnn.py --rebuild-model
```

Script sẽ đọc log, lọc dữ liệu hợp lệ, tách train/validation, tăng dữ liệu bằng đối xứng bàn cờ, train CNN và lưu model vào:

```txt
data/models/caro_supervised.h5
```

Có thể điều chỉnh quá trình train bằng các tham số CLI:

```powershell
python ai\train_cnn.py --rebuild-model --epochs 30 --batch-size 32 --learning-rate 0.001 --seed 42
```

Nếu muốn train thử trên một tập log mà không di chuyển file sau khi train, thêm:

```powershell
python ai\train_cnn.py --logs-dir data\logs_archive --rebuild-model --no-archive
```

Tham số `--no-archive` hữu ích khi dùng lại dữ liệu cũ hoặc muốn thử nghiệm nhiều lần. Nếu `--logs-dir` đã trỏ tới `data/logs_archive`, script cũng tự bỏ qua bước archive để tránh đổi tên hoặc di chuyển lại dữ liệu đã lưu trữ.

Trong lúc train, script dùng shuffle có seed cố định, EarlyStopping, ReduceLROnPlateau và checkpoint model tốt nhất. Checkpoint tốt nhất được lưu tại:

```txt
data/models/caro_supervised_best.keras
```

Lịch sử train được lưu vào:

```txt
data/training_history/cnn_training_history.json
data/training_history/cnn_training_history.csv
data/training_history/cnn_training_history.png
```

File PNG chỉ được tạo nếu môi trường có `matplotlib`. Nếu không có, JSON/CSV vẫn đủ để theo dõi loss, accuracy, top-3 accuracy và validation.

Mặc định, các log đã dùng để train sẽ được chuyển sang:

```txt
data/logs_archive/
```

## Đánh giá AI tự động

Dùng script benchmark để đo chất lượng AI mà không cần mở giao diện Pygame:

```powershell
python ai\evaluate.py --games 6
```

Benchmark mặc định chạy các cặp:

- Minimax depth 3 với Random.
- Random với Minimax depth 3.

Nếu muốn chạy thêm cặp Minimax depth 3 với Minimax depth 2:

```powershell
python ai\evaluate.py --games 3 --include-depth-suite
```

Có thể chỉ định cặp cụ thể:

```powershell
python ai\evaluate.py --games 10 --p1 minimax:3 --p2 random
```

Nếu muốn đưa CNN vào benchmark:

```powershell
python ai\evaluate.py --games 4 --include-cnn
```

Kết quả gồm số ván thắng/hòa, số nước trung bình, thời gian suy nghĩ trung bình và số nước không hợp lệ phải fallback.

## Chạy test

Chạy toàn bộ test:

```powershell
python -m unittest tests.test_logic
```

Kết quả mong đợi:

```txt
Ran 41 tests ... OK
```

Bộ test hiện kiểm tra:

- Nước đi hợp lệ và không hợp lệ.
- Điều kiện thắng ngang, dọc, chéo.
- Trường hợp 5 quân trở lên.
- Hành vi cơ bản của Minimax.
- Sinh nước đi ứng viên.
- Zobrist Hashing, quiescence search, pattern heuristic và double threat của Minimax.
- Script benchmark không cần Pygame UI.
- Kiểm tra dữ liệu train CNN.
- Chuẩn hóa bàn cờ theo góc nhìn người chơi.
- Mã hóa CNN 3 kênh, residual block và tương thích model scalar cũ.

## Hạn chế hiện tại

- Model CNN hiện có nhưng chất lượng vẫn phụ thuộc vào lượng và chất lượng log huấn luyện.
- Benchmark hiện in bảng ra terminal, chưa xuất CSV/Elo rating.
- Minimax đang dùng độ sâu 3 với Alpha-Beta, Zobrist Hashing, quiescence search, move ordering và giới hạn nhánh để giữ tốc độ phản hồi tốt.
- Heuristic chưa nhận diện đầy đủ các thế cờ nâng cao.
- CNN vẫn là supervised imitation learning nên chất lượng phụ thuộc vào dữ liệu huấn luyện.

## Hướng phát triển

- Cải thiện heuristic cho Minimax bằng bitboard hoặc lookup table sâu hơn.
- Tiếp tục cải thiện move ordering để nhận diện thêm các thế cờ phức tạp.
- Train CNN với nhiều dữ liệu hơn.
- Xuất kết quả benchmark ra CSV và tính Elo tương đối giữa các agent.
- Bổ sung thêm dữ liệu chất lượng cao để cải thiện độ chính xác của CNN.
- Thêm ảnh hoặc GIF demo vào README.
- Đóng gói game thành file chạy độc lập.

## Ghi chú

Trước khi public hoặc release, nên loại bỏ các file sinh tự động như `__pycache__/`, log tạm thời hoặc model lớn nếu không muốn đưa chúng vào repository. Các file tài liệu nộp riêng như `.docx` hoặc `.pdf` không nằm trong cấu trúc dự án.

## License

Repository hiện chưa kèm license. Hãy thêm file license nếu muốn phân phối hoặc cho phép người khác tái sử dụng dự án.
