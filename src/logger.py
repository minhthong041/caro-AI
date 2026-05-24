import json
import os
import time

def save_game_log(history, winner, mode=None):
    """
    Lưu lịch sử một ván caro thành file JSON để phục vụ huấn luyện AI.

    history chứa danh sách từng nước đi với bàn cờ trước khi đánh, tọa độ
    nước đi, người chơi và nguồn tạo nước đi. winner là kết quả cuối ván
    (0 nếu hòa, 1 hoặc 2 nếu có người thắng). mode giúp các script train
    lọc đúng nguồn dữ liệu khi đọc lại log.
    """
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'human_logs')
    os.makedirs(log_dir, exist_ok=True) 

    # Dùng nanosecond timestamp để nhiều ván liên tiếp không ghi đè file log.
    filename = f"game_{time.time_ns()}.json"
    filepath = os.path.join(log_dir, filename)

    data = {
        "winner": winner,
        "mode": mode,
        "total_moves": len(history),
        "history": history
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)
