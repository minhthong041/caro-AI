import random
import time

from src.config import BOARD_SIZE, is_winning_count


class TimeLimitExceeded(Exception):
    pass

class MinimaxAI:
    WIN_SCORE = 10_000_000
    VALID_PLAYERS = (1, 2)
    DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]
    ZOBRIST_SEED = 0xC0DEC0DE
    PATTERN_SCORES = {
        "open_four": 250_000,
        "blocked_four": 25_000,
        "open_three": 8_000,
        "broken_three": 5_000,
        "blocked_three": 500,
        "double_threat": 120_000,
    }
    PATTERN_GROUPS = {
        "open_four": (
            ".XXXX.",
            ".XXX.X.",
            ".XX.XX.",
            ".X.XXX.",
        ),
        "blocked_four": (
            "OXXXX.", ".XXXXO",
            "OXXX.X.", ".XXX.XO",
            "OXX.XX.", ".XX.XXO",
            "OX.XXX.", ".X.XXXO",
        ),
        "open_three": (
            ".XXX.",
        ),
        "broken_three": (
            ".XX.X.",
            ".X.XX.",
        ),
        "blocked_three": (
            "OXXX.", ".XXXO",
            "OXX.X.", ".XX.XO",
            "OX.XX.", ".X.XXO",
        ),
    }
    PATTERN_CACHE_LIMIT = 12_000
    
    def __init__(self, ai_player, human_player, depth=3, time_limit=1.0):
        """
        Khởi tạo AI Minimax với vai trò quân cờ và độ sâu tìm kiếm.

        ai_player là quân do máy điều khiển, human_player là quân đối thủ.
        depth là độ sâu tối đa của iterative deepening. time_limit giới hạn
        số giây AI được suy nghĩ cho mỗi nước đi.
        """
        self.ai_player = ai_player
        self.human_player = human_player
        self.depth = depth
        self.time_limit = time_limit
        self.board_size = BOARD_SIZE
        self._transposition = {}
        self._deadline = None
        self._zobrist_table = self.build_zobrist_table()
        self._pattern_cache = {}
        self.max_root_candidates = 30
        self.max_branch_candidates = 18
        self.max_double_threat_candidates = 40
        self.max_quiescence_depth = 2
        self.max_quiescence_moves = 8

    def is_valid_player(self, player):
        """
        Kiểm tra mã người chơi có thuộc hai phe hợp lệ của bàn cờ không.

        Minimax chỉ làm việc với hai giá trị quân 1 và 2. Việc chặn sớm các
        mã khác giúp các hàm thắng/thua không vô tình xem ô trống hoặc dữ
        liệu lỗi như một người chơi hợp lệ.
        """
        return player in self.VALID_PLAYERS

    def board_key(self, grid):
        """
        Tạo khóa Zobrist 64-bit cho transposition table từ ma trận bàn cờ.

        Khác với tuple lồng nhau, Zobrist Hashing chỉ cần XOR các giá trị đã
        được khởi tạo sẵn cho từng ô/quân. Trong cây Minimax, khóa này còn có
        thể cập nhật gia tăng sau mỗi nước giả lập thay vì duyệt lại cả bàn.
        """
        zobrist_hash = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                zobrist_hash ^= self.zobrist_piece(r, c, grid[r][c])
        return zobrist_hash

    def build_zobrist_table(self):
        """
        Khởi tạo bảng Zobrist quyết định được cho từng ô và từng loại quân.

        Dùng Random riêng để không làm thay đổi luồng random chọn nước đi khi
        nhiều nước có cùng điểm.
        """
        rng = random.Random(self.ZOBRIST_SEED + self.board_size)
        return [
            [
                [rng.getrandbits(64) for _ in self.VALID_PLAYERS]
                for _ in range(self.board_size)
            ]
            for _ in range(self.board_size)
        ]

    def zobrist_piece(self, row, col, player):
        """
        Trả về giá trị Zobrist của một quân tại một ô, hoặc 0 với ô trống.
        """
        if player not in self.VALID_PLAYERS:
            return 0
        return self._zobrist_table[row][col][player - 1]

    def update_board_hash(self, hash_key, row, col, player):
        """
        Cập nhật hash khi đặt hoặc xóa một quân bằng phép XOR.
        """
        return hash_key ^ self.zobrist_piece(row, col, player)

    def opponent_of(self, player):
        """
        Trả về mã quân của đối thủ tương ứng với player.

        Hàm giả định caller đã truyền player hợp lệ, vì vậy logic chỉ cần đổi
        qua lại giữa 1 và 2. Các hàm public sẽ kiểm tra tính hợp lệ trước khi
        gọi nếu dữ liệu có thể đến từ bên ngoài.
        """
        return 2 if player == 1 else 1

    def start_search_timer(self):
        """
        Khởi tạo deadline cho một lượt tìm kiếm.

        Nếu time_limit là None hoặc <= 0, AI sẽ duyệt đủ depth như cũ. Khi có
        deadline, các node trong Minimax sẽ kiểm tra định kỳ và dừng mềm.
        """
        if self.time_limit is None or self.time_limit <= 0:
            self._deadline = None
        else:
            self._deadline = time.perf_counter() + self.time_limit

    def check_time(self):
        """
        Dừng tìm kiếm khi vượt quá time budget hiện tại.
        """
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            raise TimeLimitExceeded

    def get_best_move(self, grid):
        """
        Chọn nước đi tốt nhất hiện tại bằng iterative deepening Minimax.

        AI thử lần lượt depth 1, 2, ... đến depth tối đa. Nếu hết time budget,
        hàm trả về nước tốt nhất ở độ sâu gần nhất đã hoàn tất.
        """
        self._transposition = {}
        self.start_search_timer()
        
        # Chỉ xét các nước đi gần quân đã có để giảm hệ số phân nhánh.
        moves = self.limit_root_moves(self.order_moves(grid, self.generate_moves(grid), self.ai_player))
        if not moves:
            return (self.board_size // 2, self.board_size // 2)

        immediate_win = self.find_immediate_move(grid, moves, self.ai_player)
        if immediate_win is not None:
            return immediate_win

        immediate_block = self.find_immediate_move(grid, moves, self.human_player)
        if immediate_block is not None:
            return immediate_block

        root_hash = self.board_key(grid)
        completed_best_moves = [moves[0]]
        completed_best_score = -float('inf')

        for current_depth in range(1, self.depth + 1):
            best_score = -float('inf')
            best_moves = []

            try:
                for (r, c) in moves:
                    self.check_time()
                    grid[r][c] = self.ai_player
                    child_hash = self.update_board_hash(root_hash, r, c, self.ai_player)
                    try:
                        # Ưu tiên nước thắng trực tiếp trước khi đánh giá heuristic.
                        if self.has_winner_from_move(grid, r, c, self.ai_player):
                            score = self.WIN_SCORE
                        else:
                            score = self.minimax(
                                grid,
                                current_depth - 1,
                                -float('inf'),
                                float('inf'),
                                False,
                                last_move=(r, c),
                                last_player=self.ai_player,
                                hash_key=child_hash,
                            )
                    finally:
                        grid[r][c] = 0
                    
                    if score > best_score:
                        best_score = score
                        best_moves = [(r, c)]
                    elif score == best_score:
                        best_moves.append((r, c))
            except TimeLimitExceeded:
                break

            if best_moves:
                completed_best_moves = best_moves
                completed_best_score = best_score
                if completed_best_score >= self.WIN_SCORE:
                    break
                 
        # Chọn ngẫu nhiên giữa các nước cùng điểm để tránh lối chơi lặp lại.
        return random.choice(completed_best_moves) if completed_best_moves else moves[0]

    def terminal_score(self, grid, last_move=None, last_player=None, depth_bonus=0):
        """
        Trả về điểm thắng/thua nếu trạng thái hiện tại đã kết thúc.

        Khi biết nước vừa đánh, chỉ cần kiểm tra cục bộ quanh nước đó; nhánh
        quét toàn bàn được giữ cho các caller cũ không truyền last_move.
        """
        if last_move is not None and last_player is not None:
            last_r, last_c = last_move
            if self.has_winner_from_move(grid, last_r, last_c, last_player):
                return (
                    self.WIN_SCORE + depth_bonus
                    if last_player == self.ai_player
                    else -self.WIN_SCORE - depth_bonus
                )
            return None

        if self.has_winner(grid, self.ai_player):
            return self.WIN_SCORE + depth_bonus
        if self.has_winner(grid, self.human_player):
            return -self.WIN_SCORE - depth_bonus
        return None

    def minimax(self, grid, depth, alpha, beta, is_maximizing, last_move=None, last_player=None, hash_key=None):
        """
        Duyệt cây nước đi bằng Minimax kết hợp cắt tỉa Alpha-Beta.

        is_maximizing cho biết lượt hiện tại là của AI hay đối thủ. alpha và
        beta giữ biên điểm tốt nhất đã biết để bỏ qua các nhánh chắc chắn
        không thể cải thiện kết quả.
        """
        self.check_time()
        if hash_key is None:
            hash_key = self.board_key(grid)

        cache_key = ("minimax", depth, is_maximizing, hash_key)
        if cache_key in self._transposition:
            return self._transposition[cache_key]

        # Nhận diện trạng thái kết thúc trước khi dùng heuristic tĩnh.
        score = self.terminal_score(grid, last_move, last_player, depth)
        if score is not None:
            self._transposition[cache_key] = score
            return score
        
        if depth == 0:
            return self.quiescence_search(
                grid,
                alpha,
                beta,
                is_maximizing,
                hash_key,
                self.max_quiescence_depth,
                last_move=last_move,
                last_player=last_player,
            )

        current_player = self.ai_player if is_maximizing else self.human_player
        moves = self.order_moves(grid, self.generate_moves(grid), current_player)
        moves = self.limit_moves(moves, depth)
        if not moves:
            score = self.evaluate_board(grid)
            self._transposition[cache_key] = score
            return score

        if is_maximizing:
            max_eval = -float('inf')
            had_cutoff = False
            for (r, c) in moves:
                self.check_time()
                grid[r][c] = self.ai_player
                child_hash = self.update_board_hash(hash_key, r, c, self.ai_player)
                try:
                    eval = self.minimax(
                        grid,
                        depth - 1,
                        alpha,
                        beta,
                        False,
                        last_move=(r, c),
                        last_player=self.ai_player,
                        hash_key=child_hash,
                    )
                finally:
                    grid[r][c] = 0
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    had_cutoff = True
                    break
            if not had_cutoff:
                self._transposition[cache_key] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            had_cutoff = False
            for (r, c) in moves:
                self.check_time()
                grid[r][c] = self.human_player
                child_hash = self.update_board_hash(hash_key, r, c, self.human_player)
                try:
                    eval = self.minimax(
                        grid,
                        depth - 1,
                        alpha,
                        beta,
                        True,
                        last_move=(r, c),
                        last_player=self.human_player,
                        hash_key=child_hash,
                    )
                finally:
                    grid[r][c] = 0
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    had_cutoff = True
                    break
            if not had_cutoff:
                self._transposition[cache_key] = min_eval
            return min_eval

    def count_move_patterns(self, grid, row, col, player):
        """
        Đếm pattern chiến thuật trên bốn đường đi qua một nước vừa đặt.
        """
        counts = {name: 0 for name in self.PATTERN_GROUPS}
        for dr, dc in self.DIRECTIONS:
            line = self.get_local_line_string(grid, row, col, player, dr, dc)
            line_counts = self.count_line_patterns(line)
            for name, count in line_counts.items():
                counts[name] += count
        return counts

    def score_forcing_move(self, grid, row, col, player):
        """
        Chấm điểm một nước forcing dùng riêng cho quiescence search.

        Các nước thắng ngay, tạo bốn quân, hoặc tạo nhiều mối đe dọa mở được
        xem là trạng thái "ồn" cần nhìn thêm thay vì dừng ở heuristic tĩnh.
        """
        if grid[row][col] != 0 or not self.is_valid_player(player):
            return 0

        grid[row][col] = player
        try:
            if self.has_winner_from_move(grid, row, col, player):
                return self.WIN_SCORE

            counts = self.count_move_patterns(grid, row, col, player)
            score = (
                counts["open_four"] * self.PATTERN_SCORES["open_four"]
                + counts["blocked_four"] * self.PATTERN_SCORES["blocked_four"]
                + counts["open_three"] * self.PATTERN_SCORES["open_three"]
                + counts["broken_three"] * self.PATTERN_SCORES["broken_three"]
            )
            if self.count_threat_directions_from_move(grid, row, col, player) >= 2:
                score += self.PATTERN_SCORES["double_threat"]
            return score
        finally:
            grid[row][col] = 0

    def generate_quiescence_moves(self, grid, current_player):
        """
        Chỉ sinh các nước chiến thuật đáng mở rộng ở tầng quiescence.
        """
        if not self.is_valid_player(current_player):
            return []

        opponent = self.opponent_of(current_player)
        threshold = self.PATTERN_SCORES["broken_three"]
        scored_moves = []

        for r, c in self.order_moves(grid, self.generate_moves(grid), current_player):
            self.check_time()
            attack_score = self.score_forcing_move(grid, r, c, current_player)
            block_score = self.score_forcing_move(grid, r, c, opponent) * 0.95
            forcing_score = max(attack_score, block_score)
            if forcing_score >= threshold:
                scored_moves.append((forcing_score, (r, c)))

        scored_moves.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        return [move for _score, move in scored_moves[:self.max_quiescence_moves]]

    def quiescence_search(self, grid, alpha, beta, is_maximizing, hash_key, q_depth, last_move=None, last_player=None):
        """
        Mở rộng có kiểm soát các vị trí còn đe dọa mạnh khi Minimax hết depth.

        Nếu bàn đang có nước thắng/chặn/threat rõ ràng, hàm nhìn thêm vài ply
        thay vì gọi evaluate_board ngay, giúp giảm horizon effect.
        """
        self.check_time()
        score = self.terminal_score(grid, last_move, last_player, q_depth)
        if score is not None:
            return score

        cache_key = ("quiescence", q_depth, is_maximizing, hash_key)
        if cache_key in self._transposition:
            return self._transposition[cache_key]

        stand_pat = self.evaluate_board(grid)
        if q_depth <= 0:
            self._transposition[cache_key] = stand_pat
            return stand_pat

        current_player = self.ai_player if is_maximizing else self.human_player
        moves = self.generate_quiescence_moves(grid, current_player)
        if not moves:
            self._transposition[cache_key] = stand_pat
            return stand_pat

        if is_maximizing:
            value = stand_pat
            if value >= beta:
                return value
            alpha = max(alpha, value)

            had_cutoff = False
            for r, c in moves:
                grid[r][c] = self.ai_player
                child_hash = self.update_board_hash(hash_key, r, c, self.ai_player)
                try:
                    child_score = self.quiescence_search(
                        grid,
                        alpha,
                        beta,
                        False,
                        child_hash,
                        q_depth - 1,
                        last_move=(r, c),
                        last_player=self.ai_player,
                    )
                finally:
                    grid[r][c] = 0

                value = max(value, child_score)
                alpha = max(alpha, value)
                if beta <= alpha:
                    had_cutoff = True
                    break

            if not had_cutoff:
                self._transposition[cache_key] = value
            return value

        value = stand_pat
        if value <= alpha:
            return value
        beta = min(beta, value)

        had_cutoff = False
        for r, c in moves:
            grid[r][c] = self.human_player
            child_hash = self.update_board_hash(hash_key, r, c, self.human_player)
            try:
                child_score = self.quiescence_search(
                    grid,
                    alpha,
                    beta,
                    True,
                    child_hash,
                    q_depth - 1,
                    last_move=(r, c),
                    last_player=self.human_player,
                )
            finally:
                grid[r][c] = 0

            value = min(value, child_score)
            beta = min(beta, value)
            if beta <= alpha:
                had_cutoff = True
                break

        if not had_cutoff:
            self._transposition[cache_key] = value
        return value

    def evaluate_board(self, grid):
        """
        Chấm điểm toàn bộ bàn cờ theo lợi thế hiện tại của AI.

        Điểm của AI được cộng, điểm của người chơi được trừ với trọng số
        phòng thủ cao hơn để AI ưu tiên chặn các chuỗi nguy hiểm. Ngoài chuỗi
        liên tiếp cơ bản, hàm cộng thêm điểm pattern như open four, broken
        three và double threat.
        """
        self.check_time()
        ai_score = 0
        human_score = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if grid[r][c] == self.ai_player:
                    ai_score += self.evaluate_directions(grid, r, c, self.ai_player)
                elif grid[r][c] == self.human_player:
                    human_score += self.evaluate_directions(grid, r, c, self.human_player)
                     
        # AI ưu tiên phòng ngự chặn người chơi trước khi tính đường thắng
        ai_threats = self.evaluate_threat_patterns(grid, self.ai_player)
        human_threats = self.evaluate_threat_patterns(grid, self.human_player)
        ai_double_threats = self.evaluate_double_threats(grid, self.ai_player)
        human_double_threats = self.evaluate_double_threats(grid, self.human_player)
        return (ai_score + ai_threats + ai_double_threats) - ((human_score + human_threats + human_double_threats) * 10.0)

    def count_overlapping(self, text, pattern):
        """
        Đếm số lần pattern xuất hiện, cho phép các vị trí chồng lấn nhau.
        """
        if len(pattern) > len(text):
            return 0
        count = 0
        start = text.find(pattern)
        while start != -1:
            count += 1
            start = text.find(pattern, start + 1)
        return count

    def count_line_patterns(self, line):
        """
        Đếm các thế cờ quan trọng trong một dòng đã mã hóa.

        Ký hiệu dòng: X là quân đang xét, O là quân đối thủ hoặc biên bàn cờ,
        dấu chấm là ô trống.
        """
        padded = f"O{line}O"
        cached = self._pattern_cache.get(padded)
        if cached is not None:
            return dict(cached)

        counts = {
            name: sum(self.count_overlapping(padded, pattern) for pattern in patterns)
            for name, patterns in self.PATTERN_GROUPS.items()
        }
        if len(self._pattern_cache) >= self.PATTERN_CACHE_LIMIT:
            self._pattern_cache.clear()
        self._pattern_cache[padded] = counts
        return dict(counts)

    def score_pattern_counts(self, counts):
        """
        Quy đổi số lượng pattern thành điểm heuristic.
        """
        return sum(self.PATTERN_SCORES[name] * count for name, count in counts.items())

    def build_line_string(self, cells, player):
        """
        Mã hóa một dòng bàn cờ theo góc nhìn của player.
        """
        opponent = self.opponent_of(player)
        chars = []
        for cell in cells:
            if cell == player:
                chars.append("X")
            elif cell == opponent:
                chars.append("O")
            else:
                chars.append(".")
        return "".join(chars)

    def iter_pattern_lines(self, grid, player):
        """
        Sinh mọi hàng, cột và đường chéo đủ dài để xét pattern 5 quân.
        """
        for r in range(self.board_size):
            yield self.build_line_string(grid[r], player)

        for c in range(self.board_size):
            yield self.build_line_string([grid[r][c] for r in range(self.board_size)], player)

        for start_c in range(self.board_size):
            line = []
            r, c = 0, start_c
            while r < self.board_size and c < self.board_size:
                line.append(grid[r][c])
                r += 1
                c += 1
            if len(line) >= 5:
                yield self.build_line_string(line, player)

        for start_r in range(1, self.board_size):
            line = []
            r, c = start_r, 0
            while r < self.board_size and c < self.board_size:
                line.append(grid[r][c])
                r += 1
                c += 1
            if len(line) >= 5:
                yield self.build_line_string(line, player)

        for start_c in range(self.board_size):
            line = []
            r, c = 0, start_c
            while r < self.board_size and c >= 0:
                line.append(grid[r][c])
                r += 1
                c -= 1
            if len(line) >= 5:
                yield self.build_line_string(line, player)

        for start_r in range(1, self.board_size):
            line = []
            r, c = start_r, self.board_size - 1
            while r < self.board_size and c >= 0:
                line.append(grid[r][c])
                r += 1
                c -= 1
            if len(line) >= 5:
                yield self.build_line_string(line, player)

    def evaluate_threat_patterns(self, grid, player):
        """
        Chấm điểm các pattern chiến thuật nâng cao trên toàn bàn.
        """
        if not self.is_valid_player(player):
            return 0

        score = 0
        for line in self.iter_pattern_lines(grid, player):
            self.check_time()
            score += self.score_pattern_counts(self.count_line_patterns(line))
        return score

    def get_local_line_string(self, grid, row, col, player, dr, dc, radius=5):
        """
        Lấy chuỗi cục bộ quanh một nước đi theo một hướng.
        """
        opponent = self.opponent_of(player)
        chars = []
        for offset in range(-radius, radius + 1):
            r = row + dr * offset
            c = col + dc * offset
            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                chars.append("O")
            elif grid[r][c] == player:
                chars.append("X")
            elif grid[r][c] == opponent:
                chars.append("O")
            else:
                chars.append(".")
        return "".join(chars)

    def count_threat_directions_from_move(self, grid, row, col, player):
        """
        Đếm số hướng mà nước vừa thử tạo ra mối đe dọa đáng kể.
        """
        threat_directions = 0
        for dr, dc in self.DIRECTIONS:
            line = self.get_local_line_string(grid, row, col, player, dr, dc)
            counts = self.count_line_patterns(line)
            if (
                counts["open_four"]
                or counts["blocked_four"]
                or counts["open_three"]
                or counts["broken_three"]
            ):
                threat_directions += 1
        return threat_directions

    def limit_threat_candidate_moves(self, grid, moves):
        """
        Giới hạn số candidate dùng riêng cho phát hiện double threat.
        """
        center = self.board_size // 2

        def move_key(move):
            r, c = move
            neighbors = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] != 0:
                        neighbors += 1
            center_distance = max(abs(r - center), abs(c - center))
            return (-neighbors, center_distance, r, c)

        return sorted(moves, key=move_key)[:self.max_double_threat_candidates]

    def evaluate_double_threats(self, grid, player):
        """
        Chấm điểm các ô có thể tạo hai mối đe dọa cùng lúc.

        Double threat thường buộc đối thủ không thể chặn hết trong một nước, ví
        dụ một nước vừa tạo open three ngang vừa tạo open three dọc.
        """
        if not self.is_valid_player(player):
            return 0

        score = 0
        moves = self.limit_threat_candidate_moves(grid, self.generate_moves(grid))
        for r, c in moves:
            self.check_time()
            grid[r][c] = player
            try:
                if self.count_threat_directions_from_move(grid, r, c, player) >= 2:
                    score += self.PATTERN_SCORES["double_threat"]
            finally:
                grid[r][c] = 0
        return score

    def evaluate_directions(self, grid, r, c, player):
        """
        Chấm điểm các chuỗi quân bắt đầu từ một ô theo 4 hướng chính.

        Hàm bỏ qua những ô không phải đầu chuỗi để tránh đếm trùng, sau đó
        đánh giá độ dài chuỗi và số đầu mở nhằm nhận diện các thế 2, 3, 4
        quân còn khả năng phát triển.
        """
        score = 0
        directions = self.DIRECTIONS
        
        for dr, dc in directions:
            # Chỉ chấm từ đầu chuỗi để tránh đếm trùng cùng một thế cờ.
            prev_r, prev_c = r - dr, c - dc
            if 0 <= prev_r < self.board_size and 0 <= prev_c < self.board_size:
                if grid[prev_r][prev_c] == player:
                    continue
            
            count = 1
            open_ends = 0
            
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == player:
                count += 1
                nr += dr
                nc += dc
                
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == 0:
                open_ends += 1

            if 0 <= prev_r < self.board_size and 0 <= prev_c < self.board_size and grid[prev_r][prev_c] == 0:
                open_ends += 1

            if is_winning_count(count):
                score += 100000
            elif count == 4:
                if open_ends == 2:
                    score += 10000
                elif open_ends == 1:
                    score += 1000
            elif count == 3:
                if open_ends == 2:
                    score += 1000
                elif open_ends == 1:
                    score += 100
            elif count == 2:
                if open_ends == 2:
                    score += 100
                elif open_ends == 1:
                    score += 10
                
        return score
    
    def has_winner(self, grid, player):
        """
        Kiểm tra một người chơi đã có 5 quân liên tiếp trên bàn giả lập chưa.

        Hàm được dùng trong cả tìm kiếm Minimax và các bước kiểm tra chiến
        thuật nhanh, nên chỉ trả về True/False mà không thay đổi bàn cờ.
        """
        if not self.is_valid_player(player):
            return False

        for r in range(self.board_size):
            for c in range(self.board_size):
                if grid[r][c] != player:
                    continue
                     
                for dr, dc in self.DIRECTIONS:
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < self.board_size and 0 <= prev_c < self.board_size and grid[prev_r][prev_c] == player:
                        continue
                        
                    count = 1
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == player:
                        count += 1
                        if is_winning_count(count):
                            return True
                        nr += dr
                        nc += dc
                         
        return False

    def has_winner_from_move(self, grid, row, col, player):
        """
        Kiểm tra chiến thắng cục bộ quanh nước vừa đánh.

        Khi Minimax vừa thử một nước, chỉ các chuỗi đi qua ô đó có thể thay
        đổi kết quả thắng/thua. Cách kiểm tra cục bộ này nhanh hơn nhiều so
        với việc quét lại toàn bộ bàn ở mọi node tìm kiếm.
        """
        if not self.is_valid_player(player):
            return False
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        if grid[row][col] != player:
            return False

        for dr, dc in self.DIRECTIONS:
            count = 1

            nr, nc = row + dr, col + dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == player:
                count += 1
                nr += dr
                nc += dc

            nr, nc = row - dr, col - dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == player:
                count += 1
                nr -= dr
                nc -= dc

            if is_winning_count(count):
                return True

        return False

    def generate_moves(self, grid):
        """
        Sinh danh sách nước đi ứng viên gần các quân đã xuất hiện.

        Với caro, các ô ở xa toàn bộ quân hiện có thường ít giá trị chiến
        thuật. Giới hạn ứng viên trong vùng lân cận giúp Minimax chạy nhanh
        hơn nhưng vẫn giữ được các nước tấn công/phòng thủ quan trọng.
        """
        moves = set()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if grid[r][c] != 0:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == 0:
                                moves.add((nr, nc))
        return list(moves)

    def limit_moves(self, moves, depth):
        """
        Giới hạn số nước được duyệt ở các tầng sâu của cây Minimax.

        Danh sách moves đã được sắp theo độ nguy hiểm trước đó, nên việc cắt
        bớt phần cuối giúp giảm bùng nổ nhánh trong khi vẫn giữ các lựa chọn
        chiến thuật quan trọng nhất.
        """
        if depth >= 2 and len(moves) > self.max_branch_candidates:
            return moves[:self.max_branch_candidates]
        return moves

    def limit_root_moves(self, moves):
        """
        Giới hạn số ứng viên được xét ở tầng gốc của lượt AI.

        Tầng gốc quyết định nước đi thật sự nên được phép xét nhiều ứng viên
        hơn các tầng sâu, nhưng vẫn cần giới hạn để UI không bị treo khi bàn
        cờ đã có nhiều quân.
        """
        if len(moves) > self.max_root_candidates:
            return moves[:self.max_root_candidates]
        return moves

    def find_immediate_move(self, grid, moves, player):
        """
        Tìm nước trong danh sách ứng viên giúp player thắng ngay.

        Hàm đặt thử từng ứng viên, kiểm tra thắng cục bộ rồi hoàn tác bàn cờ.
        Nếu có nước kết thúc ván lập tức, caller có thể ưu tiên nó trước khi
        chạy Minimax sâu hơn.
        """
        for r, c in moves:
            grid[r][c] = player
            is_win = self.has_winner_from_move(grid, r, c, player)
            grid[r][c] = 0
            if is_win:
                return (r, c)
        return None

    def score_line_from_move(self, grid, row, col, player):
        """
        Chấm nhanh sức mạnh chiến thuật của một nước theo các đường đi qua nó.

        Điểm được tính từ độ dài chuỗi và số đầu mở ở bốn hướng chính. Hàm
        dùng cho move ordering nên ưu tiên tốc độ và khả năng nhận diện đe
        dọa, không thay thế EvaluateBoard toàn cục.
        """
        score = 0

        for dr, dc in self.DIRECTIONS:
            count = 1
            open_ends = 0

            nr, nc = row + dr, col + dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == player:
                count += 1
                nr += dr
                nc += dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == 0:
                open_ends += 1

            nr, nc = row - dr, col - dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == player:
                count += 1
                nr -= dr
                nc -= dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] == 0:
                open_ends += 1

            if is_winning_count(count):
                score += self.WIN_SCORE
            elif count == 4:
                score += 100000 if open_ends == 2 else 10000 if open_ends == 1 else 0
            elif count == 3:
                score += 5000 if open_ends == 2 else 500 if open_ends == 1 else 0
            elif count == 2:
                score += 200 if open_ends == 2 else 20 if open_ends == 1 else 0

        return score

    def score_move_for_ordering(self, grid, move, player):
        """
        Chấm nhanh một candidate để Alpha-Beta gặp nước mạnh/nguy hiểm trước.

        Hàm kết hợp điểm tấn công của chính player và điểm phòng thủ khi giả
        lập đối thủ đánh vào cùng ô. Nhờ vậy thứ tự duyệt ưu tiên cả nước
        thắng lẫn nước cần chặn.
        """
        r, c = move
        opponent = self.opponent_of(player)

        grid[r][c] = player
        own_score = self.score_line_from_move(grid, r, c, player)
        grid[r][c] = 0

        grid[r][c] = opponent
        block_score = self.score_line_from_move(grid, r, c, opponent)
        grid[r][c] = 0

        return own_score + block_score * 0.9

    def order_moves(self, grid, moves, player=None):
        """
        Sắp xếp nước đi để Alpha-Beta xét các ô giàu tương tác trước.

        Những ô gần nhiều quân đã có thường chứa đòn tấn công/phòng thủ quan
        trọng hơn, còn khoảng cách tới tâm giúp thứ tự ổn định ở khai cuộc.
        """
        if player is None:
            player = self.ai_player

        center = self.board_size // 2

        def move_key(move):
            """
            Tạo key sắp xếp cho một nước đi ứng viên.

            Key ưu tiên threat_score cao, số hàng xóm nhiều, khoảng cách tới
            tâm nhỏ, rồi mới tới tọa độ để thứ tự ổn định giữa các nước cùng
            chất lượng.
            """
            r, c = move
            neighbors = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and grid[nr][nc] != 0:
                        neighbors += 1
            threat_score = self.score_move_for_ordering(grid, move, player)
            center_distance = max(abs(r - center), abs(c - center))
            return (-threat_score, -neighbors, center_distance, r, c)

        return sorted(moves, key=move_key)
