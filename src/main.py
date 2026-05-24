import os

# Giảm log nền của Pygame/TensorFlow trên terminal.
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import pygame
import sys
import threading
from contextlib import nullcontext

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.board import Board
from src.config import BOARD_SIZE, CELL_SIZE
from src.ui import GameUI
from src.logger import save_game_log

# Tác nhân AI.
from ai.minimax import MinimaxAI

# Trạng thái màn hình.
STATE_MENU = 0
STATE_PLAYING = 1
STATE_WIN = 2
STATE_WHO_FIRST = 3
STATE_WIN_PAUSE = 4

# Chế độ chơi.
MODE_MINIMAX = "PvP Minimax"
MODE_PVP = "PvP Collect Data"
MODE_PVP_CNN = "PvP Trained CNN"

class GameController:
    def __init__(self):
        """
        Khởi tạo toàn bộ controller điều phối UI, bàn cờ và các AI.

        Controller giữ trạng thái màn hình, lượt chơi, lịch sử ván và các
        luồng suy luận AI. Đây là lớp trung tâm nối Pygame với Minimax và CNN.
        """
        self.ui = GameUI(board_size=BOARD_SIZE)
        
        # Minimax chính dùng để chơi; bản depth=1 phục vụ kiểm tra thế bắt buộc nhanh.
        self.ai_devil = MinimaxAI(ai_player=2, human_player=1, depth=3)
        self.quick_check_ai = MinimaxAI(ai_player=2, human_player=1, depth=1)
        
        # CNN được lazy-load khi người chơi chọn mode tương ứng để game vẫn mở được nếu thiếu TensorFlow/Keras.
        self.ai_trained = None
        self.cnn_load_error = None
        self.cnn_model_signature = None
        self.cnn_load_thread = None
        self.cnn_load_request_id = 0
        
        # Trạng thái ván chơi hiện tại.
        self.game_state = STATE_MENU
        self.current_mode = None

        # Trạng thái suy luận AI chạy nền.
        self.ai_is_thinking = False
        self.pending_ai_move = None
        self.pending_ai_source = None
        self.ai_error_blocked = False
        self.pending_finish_winner = None
        self.win_pause_until = 0
        self.starting_player = 1
        self.game_version = 0
        self.ai_request_id = 0
        
        # Đồng bộ các lần gọi TensorFlow/Keras trong luồng nền.
        self.tf_lock = threading.Lock()
        
        self.reset_game_logic()

    def reset_game_logic(self, starting_player=None):
        """
        Khởi tạo lại trạng thái logic cho một ván mới.

        Hàm reset bàn cờ, lượt hiện tại, người thắng, lịch sử nước đi và mọi
        biến pending của luồng AI, nhưng không thay đổi mode đang chọn.
        """
        if starting_player in (1, 2):
            self.starting_player = starting_player
        self.game_version += 1
        self.ai_request_id += 1
        self.board = Board(size=BOARD_SIZE)
        self.current_player = self.starting_player
        self.winner = None
        self.game_history = []
        self.warning_msg = ""
        self.ai_is_thinking = False
        self.pending_ai_move = None
        self.pending_ai_source = None
        self.ai_error_blocked = False
        self.pending_finish_winner = None
        self.win_pause_until = 0

    def is_ai_request_current(self, request_id, game_version, ai_player, history_length):
        """
        Kiểm tra kết quả từ thread AI còn thuộc đúng ván và đúng lượt hiện tại.

        Mỗi lần reset hoặc bắt đầu tính AI, controller tăng version/request id.
        Điều này giúp bỏ qua kết quả trả về muộn từ thread cũ, tránh đặt quân
        vào một ván hoặc một lượt đã không còn tồn tại.
        """
        return (
            request_id == self.ai_request_id
            and game_version == self.game_version
            and self.game_state == STATE_PLAYING
            and self.current_player == ai_player
            and len(self.game_history) == history_length
        )

    def get_cnn_model_path(self):
        """
        Trả về đường dẫn tuyệt đối tới model CNN mặc định của game.

        Controller dùng đường dẫn này để kiểm tra model đã tồn tại chưa, theo
        dõi thời điểm thay đổi và lazy-load SupervisedAI khi người chơi chọn
        chế độ liên quan tới CNN.
        """
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'caro_supervised.h5'))

    def get_cnn_model_signature(self):
        """
        Lấy chữ ký thay đổi của file model CNN.

        Giá trị trả về là mốc sửa đổi nanosecond của file model. Nếu file chưa
        tồn tại, hàm trả về None để controller biết cần báo người chơi train
        model trước.
        """
        try:
            return os.stat(self.get_cnn_model_path()).st_mtime_ns
        except FileNotFoundError:
            return None

    def clear_missing_cnn_model(self):
        """
        Xóa cache CNN khi file model không còn tồn tại.

        Dùng chung cho nhánh load đồng bộ và nhánh kiểm tra nhanh ở menu để
        không giữ lại model cũ sau khi người dùng xóa hoặc chưa train xong.
        """
        self.ai_trained = None
        self.cnn_load_error = None
        self.cnn_model_signature = None
        self.warning_msg = "CNN model not found. Train the bot first."

    def get_cached_trained_ai(self, model_signature=None):
        """
        Trả về CNN đã load sẵn nếu cache vẫn khớp với file model hiện tại.

        Hàm này chỉ kiểm tra cache, không import Keras/TensorFlow, nên có thể
        gọi trực tiếp trong event click mà không làm đứng giao diện.
        """
        if model_signature is None:
            model_signature = self.get_cnn_model_signature()
        if model_signature is None:
            self.clear_missing_cnn_model()
            return None
        if (
            self.ai_trained is not None
            and getattr(self.ai_trained, 'model', None) is not None
            and self.cnn_model_signature == model_signature
        ):
            return self.ai_trained
        return None

    def is_cnn_loading(self):
        """
        Kiểm tra thread nạp CNN nền còn đang chạy hay không.
        """
        return self.cnn_load_thread is not None and self.cnn_load_thread.is_alive()

    def enter_trained_cnn_mode(self):
        """
        Chuyển sang flow chọn người đi trước cho chế độ CNN đã huấn luyện.
        """
        self.current_mode = MODE_PVP_CNN
        self.game_state = STATE_WHO_FIRST
        self.warning_msg = ""

    def select_trained_cnn_mode(self):
        """
        Xử lý nút VS TRAINED CNN mà không nạp model nặng trên UI thread.
        """
        model_signature = self.get_cnn_model_signature()
        if model_signature is None:
            self.clear_missing_cnn_model()
            return

        if self.get_cached_trained_ai(model_signature) is not None:
            self.enter_trained_cnn_mode()
            return

        if self.cnn_load_error is not None and self.cnn_model_signature == model_signature:
            self.warning_msg = "CNN unavailable. Check TensorFlow/Keras install."
            return

        if self.is_cnn_loading():
            self.warning_msg = "Loading CNN model..."
            return

        self.cnn_load_request_id += 1
        load_request_id = self.cnn_load_request_id
        self.warning_msg = "Loading CNN model..."

        def load_cnn_in_background():
            ai_trained = self.get_trained_ai()
            if load_request_id != self.cnn_load_request_id:
                return
            if ai_trained is not None and getattr(ai_trained, 'model', None) is not None:
                self.enter_trained_cnn_mode()

        self.cnn_load_thread = threading.Thread(target=load_cnn_in_background, name="CnnModelLoader")
        self.cnn_load_thread.daemon = True
        self.cnn_load_thread.start()

    def get_trained_ai(self):
        """
        Nạp CNN theo nhu cầu thay vì import TensorFlow/Keras ngay khi mở game.

        Hàm cache instance SupervisedAI khi model còn cùng chữ ký file. Nếu
        model bị xóa, được tạo mới hoặc import Keras lỗi, cache và warning sẽ
        được cập nhật để UI phản hồi đúng trạng thái hiện tại.
        """
        model_signature = self.get_cnn_model_signature()
        if model_signature is None:
            self.clear_missing_cnn_model()
            return None

        cached_ai = self.get_cached_trained_ai(model_signature)
        if cached_ai is not None:
            return cached_ai

        if self.cnn_load_error is not None and self.cnn_model_signature == model_signature:
            self.warning_msg = "CNN unavailable. Check TensorFlow/Keras install."
            return None
            
        try:
            from ai.supervised import SupervisedAI
            self.ai_trained = SupervisedAI()
            self.cnn_model_signature = model_signature
            self.cnn_load_error = None
        except Exception as e:
            self.ai_trained = None
            self.cnn_load_error = str(e)
            self.cnn_model_signature = model_signature
            self.warning_msg = "CNN unavailable. Check TensorFlow/Keras install."
            print(f"[CNN] Failed to load: {e}")
            return None
            
        return self.ai_trained

    def infer_move_source(self):
        """
        Suy ra nguồn tạo nước đi hiện tại dựa trên mode và lượt chơi.

        Nhãn nguồn được ghi vào game_history để các pipeline train lọc đúng
        dữ liệu human, minimax hoặc CNN khi đọc lại log.
        """
        if self.current_mode == MODE_PVP:
            return "human"
        if self.current_mode == MODE_MINIMAX:
            return "human" if self.current_player == 1 else "minimax"
        if self.current_mode == MODE_PVP_CNN:
            return "human" if self.current_player == 1 else "cnn"
        return "unknown"
    
    def finish_game(self, winner):
        """
        Kết thúc một ván, lưu log và chuyển sang trạng thái hiển thị kết quả.

        Log được dùng lại cho pipeline huấn luyện CNN.
        """
        self.winner = winner
        self.game_state = STATE_WIN
        try:
            save_game_log(self.game_history, self.winner, self.current_mode)
        except Exception as e:
            print(f"[LOG] Failed to save game log: {e}")
            self.warning_msg = "Could not save game log."
        
    def queue_finish_game(self, winner):
        """
        Đưa ván vào trạng thái chờ kết thúc mà không khóa event loop.

        Controller giữ màn cờ thêm một khoảng ngắn để người chơi kịp thấy
        nước thắng trước khi chuyển sang màn kết quả.
        """
        self.winner = winner
        self.pending_finish_winner = winner
        self.win_pause_until = pygame.time.get_ticks() + 800
        self.game_state = STATE_WIN_PAUSE
    
    def find_forced_move(self, board_grid, ai_player, opponent_player):
        """
        Tìm nước thắng ngay hoặc nước bắt buộc phải chặn.

        Hàm thử từng ô trống cho AI trước, sau đó thử cho đối thủ. Nếu một
        nước tạo thành 5 quân liên tiếp thì trả về nước đó ngay.
        """
        valid_moves = [(r, c) for r in range(self.board.size) for c in range(self.board.size) if board_grid[r][c] == 0]
        
        for player in [ai_player, opponent_player]:
            for r, c in valid_moves:
                board_grid[r][c] = player
                is_win = self.quick_check_ai.has_winner_from_move(board_grid, r, c, player)
                board_grid[r][c] = 0
                if is_win:
                    return (r, c)
                    
        return None
    
    def get_valid_moves(self, board_grid):
        """
        Trả về mọi ô còn trống trên bàn cờ truyền vào.

        Hàm dùng cho fallback khi AI không chọn được nước đi hợp lệ. Nó không
        đọc trực tiếp self.board.grid để caller có thể truyền bản sao bàn cờ
        trong thread nền.
        """
        return [
            (r, c)
            for r in range(self.board.size)
            for c in range(self.board.size)
            if board_grid[r][c] == 0
        ]
    
    def normalize_move(self, board_grid, move):
        """
        Chuẩn hóa một nước đi AI trả về thành tuple (row, col) hợp lệ.

        Nếu AI trả về None, sai kiểu, ngoài biên hoặc ô đã bị chiếm, hàm trả
        về None để caller có thể chọn fallback thay vì làm kẹt lượt.
        """
        if not isinstance(move, (list, tuple)) or len(move) != 2:
            return None
            
        try:
            row, col = int(move[0]), int(move[1])
        except (TypeError, ValueError):
            return None
            
        if 0 <= row < self.board.size and 0 <= col < self.board.size and board_grid[row][col] == 0:
            return (row, col)
        return None
    
    def get_fallback_move(self, board_grid):
        """
        Chọn một nước hợp lệ khi AI không trả được nước đi dùng được.

        Ưu tiên vùng ứng viên quanh các quân đã có để fallback vẫn tương đối
        tự nhiên; nếu không có ứng viên thì lấy bất kỳ ô trống nào còn lại.
        """
        candidate_moves = self.ai_devil.generate_moves(board_grid)
        if candidate_moves:
            return random.choice(candidate_moves)
            
        valid_moves = self.get_valid_moves(board_grid)
        return random.choice(valid_moves) if valid_moves else None

    def execute_move(self, row, col, source=None):
        """
        Thực thi một nước cờ hợp lệ, ghi lịch sử và cập nhật kết quả ván.

        Hàm lưu bàn cờ trước khi đánh để phục vụ train, kiểm tra thắng/hòa
        ngay sau khi đặt quân và chỉ đổi lượt nếu ván vẫn tiếp tục.
        """
        
        # Lưu snapshot bàn cờ trước nước đi để phục vụ dữ liệu huấn luyện.
        board_before = [r[:] for r in self.board.grid]

        if self.board.place_piece(row, col, self.current_player):
            move_source = source or self.infer_move_source()
            self.game_history.append({"board": board_before, "move": [row, col], "player": self.current_player, "source": move_source})
            if self.board.check_win(row, col, self.current_player):
                self.queue_finish_game(self.current_player)

            elif len(self.game_history) >= self.board.size * self.board.size:
                self.queue_finish_game(0)

            else:
                self.current_player = 2 if self.current_player == 1 else 1
            return True
            
        return False

    def start_ai_thread(self, ai_agent, fallback_random=False, use_quick_check=False, source=None, pass_player=False):
        """
        Khởi chạy luồng nền để AI tính nước đi mà không làm treo giao diện.

        board_copy tách khỏi bàn cờ chính để AI có thể thử nghiệm an toàn.
        Khi luồng xong, nước đi được đưa vào pending_ai_move cho vòng lặp
        Pygame chính xử lý ở frame kế tiếp.
        """
        self.ai_request_id += 1
        request_id = self.ai_request_id
        request_game_version = self.game_version
        request_history_length = len(self.game_history)
        self.ai_is_thinking = True
        self.ai_error_blocked = False
        
        # AI chỉ đọc bản sao để không ảnh hưởng bàn cờ đang hiển thị.
        board_copy = [r[:] for r in self.board.grid]
        ai_player = self.current_player
        opponent_player = 2 if ai_player == 1 else 1
        move_source = source or self.infer_move_source()
        
        def calculate():
            """
            Tính nước đi trong thread nền và đẩy kết quả về controller.

            Hàm ưu tiên kiểm tra chiến thuật nhanh, sau đó gọi AI chính. Mọi
            exception được bắt để UI vẫn tiếp tục chạy nếu model gặp lỗi.
            """
            move = None
            move_source_for_log = move_source
            warning_msg = None
            error_blocked = False
            try:
                # Nút chiến thuật nhanh: thắng ngay nếu có, nếu không thì chặn đối thủ thắng ngay.
                if use_quick_check:
                    move = self.find_forced_move(board_copy, ai_player, opponent_player)
                        
                # Nếu check nhanh không ra kết quả nguy hiểm, dùng AI chính
                if move is None:
                    # TensorFlow/Keras không được gọi đồng thời từ nhiều thread.
                    lock_context = self.tf_lock if pass_player else nullcontext()
                    with lock_context:
                        if pass_player:
                            move = ai_agent.get_best_move(board_copy, ai_player)
                        else:
                            move = ai_agent.get_best_move(board_copy)
                
                normalized_move = self.normalize_move(board_copy, move)
                if normalized_move is None and fallback_random:
                    normalized_move = self.get_fallback_move(board_copy)
                    if normalized_move is not None:
                        move_source_for_log = "fallback"
                move = normalized_move
            except Exception as e:
                print(f"[AI] Error while calculating move: {e}")
                if fallback_random:
                    move = self.get_fallback_move(board_copy)
                    move_source_for_log = "fallback" if move is not None else None
                else:
                    move = None
                    move_source_for_log = None
                    warning_msg = "AI error. Replay or return to menu."
                    error_blocked = True
                    
            if move is None and fallback_random:
                warning_msg = "AI could not find a valid move. Replay or return to menu."
                error_blocked = True
                
            if request_id != self.ai_request_id:
                return
                
            if not self.is_ai_request_current(request_id, request_game_version, ai_player, request_history_length):
                self.ai_is_thinking = False
                return
                 
            # Lưu lại nước đi vào biến pending để luồng chính Pygame nhận diện.
            # Gán move cuối cùng rồi mới hạ cờ thinking để tránh frame chính khởi chạy thêm thread AI.
            self.pending_ai_source = move_source_for_log if move is not None else None
            if warning_msg:
                self.warning_msg = warning_msg
            self.ai_error_blocked = error_blocked
            self.pending_ai_move = move
            self.ai_is_thinking = False
             
        t = threading.Thread(target=calculate)
        t.daemon = True
        t.start()

    def handle_events(self):
        """
        Xử lý toàn bộ sự kiện Pygame trong frame hiện tại.

        Hàm điều hướng giữa các màn hình, xử lý click của người chơi, bỏ qua
        input khi AI đang suy nghĩ và trả về False khi người dùng muốn thoát.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                
                if self.ai_is_thinking:
                    continue

                if self.game_state == STATE_MENU:
                    if self.ui.rect_btn_minimax.collidepoint(pos):
                        self.cnn_load_request_id += 1
                        self.current_mode = MODE_MINIMAX
                        self.game_state = STATE_WHO_FIRST
                        self.warning_msg = ""
                    elif self.ui.rect_btn_pvp.collidepoint(pos):
                        self.cnn_load_request_id += 1
                        self.current_mode = MODE_PVP
                        self.game_state = STATE_WHO_FIRST
                        self.warning_msg = ""
                    elif self.ui.rect_btn_trained.collidepoint(pos):
                        self.select_trained_cnn_mode()
                        
                elif self.game_state == STATE_WHO_FIRST:
                    if self.ui.rect_btn_p1_first.collidepoint(pos):
                        self.reset_game_logic(starting_player=1)
                        self.game_state = STATE_PLAYING
                    elif self.ui.rect_btn_p2_first.collidepoint(pos):
                        self.reset_game_logic(starting_player=2)
                        self.game_state = STATE_PLAYING
                    elif self.ui.rect_btn_back.collidepoint(pos):
                        self.game_state = STATE_MENU
                        
                elif self.game_state == STATE_PLAYING:
                    if self.ui.rect_replay.collidepoint(pos):
                        self.reset_game_logic()
                        continue
                    elif self.ui.rect_menu.collidepoint(pos): 
                        self.game_state = STATE_MENU
                        continue
                    elif self.ui.rect_exit.collidepoint(pos): 
                        self.game_state = STATE_WHO_FIRST
                        continue
                        
                    is_human = (self.current_mode == MODE_PVP) or \
                               (self.current_player == 1 and self.current_mode in [MODE_MINIMAX, MODE_PVP_CNN])

                    if is_human:
                        x, y = pos[0] - self.ui.GRID_OFFSET_X, pos[1] - self.ui.GRID_OFFSET_Y
                        if 0 <= x < BOARD_SIZE * CELL_SIZE and 0 <= y < BOARD_SIZE * CELL_SIZE: 
                            self.execute_move(y // CELL_SIZE, x // CELL_SIZE, source="human")
                            
                elif self.game_state == STATE_WIN:
                    if self.ui.rect_replay_end.collidepoint(pos): self.game_state = STATE_WHO_FIRST
                    elif self.ui.rect_menu_end.collidepoint(pos): self.game_state = STATE_MENU
                    elif self.ui.rect_exit_end.collidepoint(pos): return False
                    
        return True

    def run(self):
        """
        Chạy vòng lặp chính của game ở 60 FPS.

        Mỗi frame xử lý input, nhận kết quả từ luồng AI, điều phối AI theo
        mode hiện tại và render đúng màn hình.
        """
        clock = pygame.time.Clock()
        while True:
            if not self.handle_events(): break
            
            last_m = tuple(self.game_history[-1]['move']) if self.game_history else None

            if self.game_state == STATE_PLAYING:
                if self.pending_ai_move is not None:
                    pending_move = self.pending_ai_move
                    pending_source = self.pending_ai_source
                    self.pending_ai_move = None
                    self.pending_ai_source = None
                    self.execute_move(pending_move[0], pending_move[1], source=pending_source)
                    
                elif not self.ai_is_thinking and not self.ai_error_blocked:
                    if self.current_player == 2 and self.current_mode == MODE_PVP_CNN:
                        ai_trained = self.get_trained_ai()
                        if ai_trained is not None and getattr(ai_trained, 'model', None) is not None:
                            self.start_ai_thread(ai_trained, fallback_random=True, use_quick_check=True, source="cnn", pass_player=True)
                        else:
                            self.warning_msg = "CNN model not found. Train the bot first."
                            self.ai_error_blocked = True
                        
                    elif self.current_player == 2 and self.current_mode == MODE_MINIMAX:
                        self.start_ai_thread(self.ai_devil, fallback_random=True, use_quick_check=True, source="minimax")

            msg = self.warning_msg
            if self.ai_is_thinking:
                msg = "AI is thinking..."
            if self.game_state == STATE_MENU:
                self.ui.draw_start_menu(msg)
            elif self.game_state == STATE_WHO_FIRST:
                self.ui.draw_who_first_screen()
            elif self.game_state == STATE_PLAYING:
                self.ui.draw_game_play(self.board.grid, self.current_player, self.current_mode, msg, last_move=last_m)
            elif self.game_state == STATE_WIN_PAUSE:
                self.ui.draw_game_play(self.board.grid, self.current_player, self.current_mode, msg, last_move=last_m)
                if pygame.time.get_ticks() >= self.win_pause_until:
                    self.finish_game(self.pending_finish_winner)
                    self.pending_finish_winner = None
            elif self.game_state == STATE_WIN:
                self.ui.draw_end_screen(self.winner, self.current_mode)
                    
            clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    GameController().run()
