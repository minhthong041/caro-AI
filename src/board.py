from src.config import BOARD_SIZE, is_winning_count


class Board:
    VALID_PLAYERS = (1, 2)

    def __init__(self, size=BOARD_SIZE):
        """
        Tạo bàn cờ caro vuông và khởi tạo toàn bộ ô ở trạng thái trống.

        size mặc định lấy từ cấu hình chung của dự án. Mỗi ô trong grid dùng
        quy ước 0 là trống, 1 là Angel/P1 và 2 là Devil/P2.
        """
        self.size = size
        # 0: ô trống, 1: người chơi 1, 2: người chơi 2.
        self.grid = [[0 for _ in range(size)] for _ in range(size)]

    def is_valid_player(self, player):
        """
        Kiểm tra mã người chơi có thuộc hai phe hợp lệ của bàn cờ không.

        Board chỉ cho phép đặt quân của player 1 hoặc 2. Việc kiểm tra này
        ngăn dữ liệu sai như 0, None hoặc số khác được ghi vào ma trận bàn cờ.
        """
        return player in self.VALID_PLAYERS

    def is_valid_move(self, row, col):
        """
        Kiểm tra một nước đi có hợp lệ hay không.

        Nước đi chỉ hợp lệ khi tọa độ nằm trong biên bàn cờ và ô đó
        vẫn còn trống, giúp các lớp điều khiển không phải lặp lại cùng
        một đoạn kiểm tra trước khi đặt quân.
        """
        return 0 <= row < self.size and 0 <= col < self.size and self.grid[row][col] == 0

    def place_piece(self, row, col, player):
        """
        Đặt quân của người chơi vào bàn cờ nếu nước đi hợp lệ.

        Hàm trả về True khi cập nhật thành công và False khi ô đã bị chiếm
        hoặc tọa độ nằm ngoài bàn cờ, để luồng game quyết định có đổi lượt
        hay bỏ qua thao tác.
        """
        if not self.is_valid_player(player):
            return False
        if self.is_valid_move(row, col):
            self.grid[row][col] = player
            return True
        return False

    def check_win(self, row, col, player):
        """
        Kiểm tra người chơi vừa đặt quân có tạo thành 5 quân liên tiếp không.

        Thuật toán chỉ tỏa ra từ nước đi mới nhất theo 4 hướng chính
        (ngang, dọc, chéo xuôi, chéo ngược), nhờ vậy nhanh hơn so với việc
        quét lại toàn bộ bàn cờ sau mỗi lượt.
        """
        if not self.is_valid_player(player):
            return False
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.grid[row][col] != player:
            return False

        # Bốn hướng cần kiểm tra: ngang, dọc và hai đường chéo.
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.grid[r][c] == player:
                count += 1
                r += dr
                c += dc
                
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.grid[r][c] == player:
                count += 1
                r -= dr
                c -= dc
                
            if is_winning_count(count):
                return True
                
        return False
