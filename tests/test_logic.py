import importlib.util
import random
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ai.evaluate import parse_agent_spec, simulate_game
from ai.minimax import MinimaxAI
from src.board import Board
from src.config import BOARD_SIZE
from src.main import GameController, MODE_PVP_CNN, STATE_MENU, STATE_PLAYING, STATE_WHO_FIRST


class BoardLogicTest(unittest.TestCase):
    def test_rejects_occupied_and_out_of_bounds_moves(self):
        """
        Kiểm tra Board từ chối ô đã có quân và tọa độ nằm ngoài biên.

        Test này bảo vệ contract của place_piece: chỉ nước hợp lệ mới thay đổi
        ma trận bàn cờ, các nước sai phải trả về False và giữ nguyên dữ liệu.
        """
        board = Board()

        self.assertTrue(board.place_piece(7, 7, 1))
        self.assertFalse(board.place_piece(7, 7, 2))
        self.assertFalse(board.place_piece(-1, 7, 1))
        self.assertFalse(board.place_piece(BOARD_SIZE, 7, 1))
        self.assertFalse(board.place_piece(7, BOARD_SIZE, 1))
        self.assertEqual(board.grid[7][7], 1)

    def test_five_wins(self):
        """
        Xác nhận 5 quân ngang liên tiếp được tính là thắng.

        Đây là trường hợp cơ bản nhất của check_win, dùng để đảm bảo luật
        WIN_LENGTH được áp dụng đúng theo hướng ngang.
        """
        board = Board()
        for col in range(5):
            self.assertTrue(board.place_piece(7, col, 1))

        self.assertTrue(board.check_win(7, 4, 1))

    def test_vertical_five_wins(self):
        """
        Xác nhận 5 quân dọc liên tiếp được tính là thắng.

        Test đảm bảo check_win quét đúng theo trục hàng và không chỉ hoạt động
        với các chuỗi nằm ngang.
        """
        board = Board()
        for row in range(3, 8):
            self.assertTrue(board.place_piece(row, 5, 2))

        self.assertTrue(board.check_win(7, 5, 2))

    def test_main_diagonal_five_wins(self):
        """
        Xác nhận đường chéo chính có 5 quân liên tiếp được tính là thắng.

        Đường chéo chính tăng đồng thời row và col, là một trong bốn hướng
        bắt buộc của luật caro.
        """
        board = Board()
        for offset in range(5):
            self.assertTrue(board.place_piece(4 + offset, 4 + offset, 1))

        self.assertTrue(board.check_win(8, 8, 1))

    def test_anti_diagonal_five_wins(self):
        """
        Xác nhận đường chéo phụ có 5 quân liên tiếp được tính là thắng.

        Đường chéo phụ tăng row và giảm col, giúp kiểm tra hướng chéo ngược
        trong thuật toán check_win.
        """
        board = Board()
        for offset in range(5):
            self.assertTrue(board.place_piece(4 + offset, 8 - offset, 2))

        self.assertTrue(board.check_win(8, 4, 2))

    def test_four_is_not_a_win(self):
        """
        Đảm bảo chuỗi 4 quân chưa đủ điều kiện thắng.

        Test này ngăn lỗi off-by-one khi so sánh số quân liên tiếp với
        WIN_LENGTH.
        """
        board = Board()
        for col in range(4):
            self.assertTrue(board.place_piece(7, col, 1))

        self.assertFalse(board.check_win(7, 3, 1))

    def test_overline_wins_with_five_or_more_rule(self):
        """
        Đảm bảo chuỗi hơn 5 quân vẫn được tính là thắng.

        Dự án đang dùng luật đủ 5 trở lên, vì vậy overline không bị xem là
        phạm luật hoặc bị loại khỏi điều kiện thắng.
        """
        board = Board()
        for col in range(6):
            self.assertTrue(board.place_piece(7, col, 1))

        self.assertTrue(board.check_win(7, 5, 1))

    def test_check_win_requires_target_cell_to_belong_to_player(self):
        """
        Đảm bảo check_win chỉ xét từ ô thuộc về đúng người chơi.

        Nếu caller truyền một tọa độ trống hoặc không phải quân của player,
        hàm phải trả về False thay vì quét nhầm từ vị trí không liên quan.
        """
        board = Board()
        for col in range(5):
            self.assertTrue(board.place_piece(7, col, 1))

        self.assertFalse(board.check_win(0, 0, 1))

    def test_rejects_invalid_player_codes(self):
        """
        Đảm bảo Board không nhận mã player ngoài 1 và 2.

        Nếu player=0 được chấp nhận, ô trống có thể bị xem như một nước đi
        thành công và check_win có nguy cơ đếm chuỗi ô trống như chuỗi thắng.
        """
        board = Board()

        self.assertFalse(board.place_piece(0, 0, 0))
        self.assertEqual(board.grid[0][0], 0)
        self.assertFalse(board.check_win(0, 0, 0))
        self.assertFalse(board.place_piece(0, 0, 3))
        self.assertEqual(board.grid[0][0], 0)


class MinimaxLogicTest(unittest.TestCase):
    def test_returns_center_on_empty_board(self):
        """
        Kiểm tra Minimax chọn ô trung tâm khi bàn cờ chưa có quân.

        Đây là fallback khai cuộc quan trọng vì generate_moves không có ứng
        viên nào khi bàn cờ hoàn toàn trống.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        ai = MinimaxAI(ai_player=2, human_player=1, depth=2)

        self.assertEqual(ai.get_best_move(grid), (BOARD_SIZE // 2, BOARD_SIZE // 2))

    def test_plays_immediate_winning_move(self):
        """
        Đảm bảo Minimax ưu tiên nước thắng ngay khi có cơ hội.

        Test tạo thế bốn quân của AI đã bị chặn một đầu, nên nước duy nhất
        còn lại phải hoàn thành chuỗi 5 quân.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        grid[7][3] = 1
        for col in range(4, 8):
            grid[7][col] = 2

        ai = MinimaxAI(ai_player=2, human_player=1, depth=2)

        self.assertEqual(ai.get_best_move(grid), (7, 8))

    def test_blocks_one_open_four(self):
        """
        Đảm bảo Minimax chặn chuỗi bốn quân hở một đầu của đối thủ.

        Đây là tình huống phòng thủ bắt buộc; AI phải chọn ô chặn thay vì
        theo đuổi một nước heuristic khác.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for col in range(4, 8):
            grid[7][col] = 1
        grid[7][3] = 2

        ai = MinimaxAI(ai_player=2, human_player=1, depth=2)

        self.assertEqual(ai.get_best_move(grid), (7, 8))

    def test_get_best_move_does_not_mutate_grid(self):
        """
        Đảm bảo get_best_move không để lại nước thử trên bàn cờ truyền vào.

        Minimax có nhiều bước giả lập đặt/xóa quân, nên test này bảo vệ việc
        hoàn tác đầy đủ sau khi tính toán.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        grid[7][7] = 1
        grid[7][8] = 2
        grid[8][8] = 1
        original_grid = [row[:] for row in grid]

        ai = MinimaxAI(ai_player=2, human_player=1, depth=2)
        ai.get_best_move(grid)

        self.assertEqual(grid, original_grid)

    def test_overline_counts_as_win(self):
        """
        Kiểm tra has_winner công nhận chuỗi hơn 5 quân là thắng.

        Luật này cần đồng nhất với Board.check_win để AI và logic game không
        hiểu khác nhau về cùng một thế cờ.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for col in range(6):
            grid[7][col] = 1

        ai = MinimaxAI(ai_player=2, human_player=1, depth=2)

        self.assertTrue(ai.has_winner(grid, 1))

    def test_has_winner_rejects_invalid_player_codes(self):
        """
        Đảm bảo Minimax không xem ô trống là quân cờ của player=0.

        player=0 là mã ô trống trong toàn bộ dự án, nên has_winner phải trả về
        False dù trên bàn có rất nhiều ô mang giá trị 0.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        ai = MinimaxAI(ai_player=2, human_player=1, depth=2)

        self.assertFalse(ai.has_winner(grid, 0))
        self.assertFalse(ai.has_winner(grid, 3))

    def test_has_winner_from_move_checks_only_local_line(self):
        """
        Đảm bảo kiểm tra thắng cục bộ trả đúng kết quả từ nước vừa đánh.

        Test tạo một chuỗi đủ thắng đi qua đúng tọa độ truyền vào để xác nhận
        has_winner_from_move hoạt động nhất quán với has_winner nhưng nhanh hơn.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for col in range(4, 9):
            grid[7][col] = 1
        ai = MinimaxAI(ai_player=2, human_player=1, depth=2)

        self.assertTrue(ai.has_winner_from_move(grid, 7, 8, 1))
        self.assertFalse(ai.has_winner_from_move(grid, 0, 0, 1))

    def test_order_moves_prioritizes_immediate_threats(self):
        """
        Đảm bảo move ordering đưa ô chặn/thắng ngay lên trước các nước thường.

        Thứ tự ứng viên ảnh hưởng trực tiếp tới hiệu quả cắt tỉa Alpha-Beta,
        vì vậy threat rõ ràng phải nằm ở đầu danh sách sau khi order_moves chạy.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for col in range(4, 8):
            grid[7][col] = 1
        ai = MinimaxAI(ai_player=2, human_player=1, depth=2)

        moves = ai.order_moves(grid, ai.generate_moves(grid), ai.ai_player)

        self.assertIn(moves[0], {(7, 3), (7, 8)})

    def test_generate_moves_uses_neighbors_only(self):
        """
        Đảm bảo generate_moves chỉ sinh các ô trống kề quân đã đánh.

        Test giữ phạm vi ứng viên quanh một quân duy nhất, giúp Minimax không
        lãng phí thời gian trên toàn bộ bàn cờ.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        grid[7][7] = 1
        ai = MinimaxAI(ai_player=2, human_player=1, depth=2)

        expected_moves = {
            (6, 6), (6, 7), (6, 8),
            (7, 6),         (7, 8),
            (8, 6), (8, 7), (8, 8),
        }

        self.assertEqual(set(ai.generate_moves(grid)), expected_moves)

    def test_zobrist_hash_updates_incrementally(self):
        """
        Đảm bảo Zobrist hash có thể cập nhật bằng XOR khi đặt/xóa quân.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        ai = MinimaxAI(ai_player=2, human_player=1, depth=1)

        empty_hash = ai.board_key(grid)
        grid[7][7] = 2
        updated_hash = ai.board_key(grid)

        self.assertEqual(updated_hash, ai.update_board_hash(empty_hash, 7, 7, 2))
        self.assertEqual(empty_hash, ai.update_board_hash(updated_hash, 7, 7, 2))

    def test_quiescence_moves_include_forced_block(self):
        """
        Đảm bảo tầng quiescence vẫn xét nước chặn khi đối thủ có open four.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for col in range(4, 8):
            grid[7][col] = 1
        ai = MinimaxAI(ai_player=2, human_player=1, depth=1)

        moves = set(ai.generate_quiescence_moves(grid, current_player=2))

        self.assertTrue({(7, 3), (7, 8)} & moves)

    def test_default_minimax_uses_depth_three(self):
        """
        Đảm bảo Minimax mặc định dùng depth 3 sau khi thêm iterative deepening.
        """
        ai = MinimaxAI(ai_player=2, human_player=1)

        self.assertEqual(ai.depth, 3)

    def test_open_four_scores_higher_than_blocked_four(self):
        """
        Kiểm tra heuristic pattern phân biệt open four và blocked four.

        Open four nguy hiểm hơn vì có hai đầu mở, nên điểm pattern phải cao
        hơn chuỗi bốn quân bị chặn một đầu.
        """
        ai = MinimaxAI(ai_player=2, human_player=1, depth=1)
        open_grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        blocked_grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for col in range(5, 9):
            open_grid[7][col] = 2
            blocked_grid[7][col] = 2
        blocked_grid[7][4] = 1

        self.assertGreater(
            ai.evaluate_threat_patterns(open_grid, 2),
            ai.evaluate_threat_patterns(blocked_grid, 2),
        )

    def test_broken_three_pattern_is_recognized(self):
        """
        Đảm bảo heuristic nhận diện thế broken three như XX_X.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        grid[7][6] = 2
        grid[7][7] = 2
        grid[7][9] = 2
        ai = MinimaxAI(ai_player=2, human_player=1, depth=1)

        counts = ai.count_line_patterns(ai.build_line_string(grid[7], 2))

        self.assertGreater(counts["broken_three"], 0)

    def test_double_threat_move_is_recognized(self):
        """
        Đảm bảo AI nhận diện một nước tạo hai open-three cùng lúc.
        """
        grid = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        grid[7][6] = 2
        grid[7][8] = 2
        grid[6][7] = 2
        grid[8][7] = 2
        ai = MinimaxAI(ai_player=2, human_player=1, depth=1)

        grid[7][7] = 2
        try:
            threats = ai.count_threat_directions_from_move(grid, 7, 7, 2)
        finally:
            grid[7][7] = 0

        self.assertGreaterEqual(threats, 2)


class EvaluateScriptTest(unittest.TestCase):
    def test_parse_minimax_depth_spec(self):
        """
        Đảm bảo script benchmark đọc đúng cấu hình minimax:depth.

        Đây là contract CLI quan trọng để có thể so sánh nhiều độ sâu khác nhau
        mà không phải sửa code trong ai/evaluate.py.
        """
        self.assertEqual(parse_agent_spec("minimax:2"), ("minimax", 2))
        self.assertEqual(parse_agent_spec("minimax"), ("minimax", 3))

    def test_random_vs_random_simulation_finishes(self):
        """
        Kiểm tra simulator benchmark chạy được một ván không cần Pygame UI.

        Test giới hạn số nước thấp để chạy nhanh, chỉ xác nhận kết quả hợp lệ
        và agent random không sinh nước đi sai.
        """
        random.seed(0)

        result = simulate_game("random", "random", max_moves=12)

        self.assertIn(result.winner, (0, 1, 2))
        self.assertLessEqual(result.total_moves, 12)
        self.assertEqual(result.invalid_moves[1], 0)
        self.assertEqual(result.invalid_moves[2], 0)


class GameControllerStateTest(unittest.TestCase):
    def make_controller_without_ui(self):
        """
        Tạo controller tối thiểu để test state machine mà không khởi tạo Pygame UI.

        Helper này dùng __new__ để bỏ qua GameController.__init__, nhờ đó test
        logic reset/cache mà không mở cửa sổ Pygame hoặc load asset đồ họa.
        """
        controller = GameController.__new__(GameController)
        controller.ai_trained = None
        controller.cnn_load_error = None
        controller.cnn_model_signature = None
        controller.cnn_load_thread = None
        controller.cnn_load_request_id = 0
        controller.starting_player = 1
        controller.game_version = 0
        controller.ai_request_id = 0
        controller.reset_game_logic()
        controller.game_state = STATE_PLAYING
        return controller

    def test_reset_preserves_selected_starting_player(self):
        """
        Đảm bảo replay ván sau giữ người đi trước đã chọn.

        Khi người chơi chọn P1 hoặc P2 đi trước, reset_game_logic không được
        tự động quay về P1 nếu không có starting_player mới được truyền vào.
        """
        controller = self.make_controller_without_ui()

        controller.reset_game_logic(starting_player=2)
        self.assertEqual(controller.current_player, 2)

        controller.reset_game_logic()
        self.assertEqual(controller.current_player, 2)

    def test_reset_invalidates_old_ai_request(self):
        """
        Đảm bảo kết quả AI từ ván cũ không còn được nhận sau khi reset.

        Test ghi lại request id và game version trước reset, sau đó xác nhận
        is_ai_request_current từ chối kết quả cũ để tránh đặt quân trễ.
        """
        controller = self.make_controller_without_ui()
        request_id = controller.ai_request_id
        game_version = controller.game_version
        ai_player = controller.current_player
        history_length = len(controller.game_history)

        self.assertTrue(controller.is_ai_request_current(request_id, game_version, ai_player, history_length))

        controller.reset_game_logic()
        controller.game_state = STATE_PLAYING

        self.assertFalse(controller.is_ai_request_current(request_id, game_version, ai_player, history_length))

    def test_missing_cnn_model_clears_stale_cached_ai(self):
        """
        Đảm bảo cache CNN cũ bị xóa khi file model không còn tồn tại.

        Trường hợp này xảy ra khi người dùng xóa model hoặc train chưa xong.
        Controller cần xóa cache và hiển thị cảnh báo thay vì giữ AI cũ.
        """
        controller = self.make_controller_without_ui()
        controller.ai_trained = type("StaleAI", (), {"model": None})()
        controller.cnn_load_error = "old error"
        controller.cnn_model_signature = 123
        controller.get_cnn_model_signature = lambda: None

        self.assertIsNone(controller.get_trained_ai())
        self.assertIsNone(controller.ai_trained)
        self.assertIsNone(controller.cnn_load_error)
        self.assertIsNone(controller.cnn_model_signature)
        self.assertEqual(controller.warning_msg, "CNN model not found. Train the bot first.")

    def test_cnn_model_reloads_when_model_file_appears(self):
        """
        Đảm bảo controller thử load lại CNN khi có chữ ký file model mới.

        Test giả lập module ai.supervised để kiểm tra nhánh reload mà không
        phụ thuộc vào TensorFlow thật hoặc file model thật trên ổ đĩa.
        """
        controller = self.make_controller_without_ui()
        controller.ai_trained = type("StaleAI", (), {"model": None})()
        controller.cnn_load_error = None
        controller.cnn_model_signature = None
        controller.get_cnn_model_signature = lambda: 456

        class FakeSupervisedAI:
            """
            Fake CNN agent có model hợp lệ để kiểm tra nhánh reload.

            Class giả lập đúng phần interface mà GameController cần: thuộc
            tính model khác None sau khi khởi tạo thành công.
            """
            model = object()

        fake_module = types.ModuleType("ai.supervised")
        fake_module.SupervisedAI = FakeSupervisedAI

        with mock.patch.dict(sys.modules, {"ai.supervised": fake_module}):
            loaded_ai = controller.get_trained_ai()

        self.assertIsInstance(loaded_ai, FakeSupervisedAI)
        self.assertIs(controller.ai_trained, loaded_ai)
        self.assertEqual(controller.cnn_model_signature, 456)
        self.assertIsNone(controller.cnn_load_error)

    def test_select_trained_cnn_mode_uses_cached_model_immediately(self):
        """
        Nếu CNN đã được load và file model chưa đổi, menu phải vào mode ngay.

        Trường hợp này bảo vệ cache để người chơi không phải chờ lại sau khi
        quay về menu rồi chọn VS TRAINED CNN lần nữa trong cùng phiên chạy.
        """
        controller = self.make_controller_without_ui()
        controller.game_state = STATE_MENU
        controller.ai_trained = type("ReadyAI", (), {"model": object()})()
        controller.cnn_model_signature = 789
        controller.get_cnn_model_signature = lambda: 789
        controller.get_trained_ai = mock.Mock(side_effect=AssertionError("should not load synchronously"))

        controller.select_trained_cnn_mode()

        controller.get_trained_ai.assert_not_called()
        self.assertEqual(controller.current_mode, MODE_PVP_CNN)
        self.assertEqual(controller.game_state, STATE_WHO_FIRST)
        self.assertEqual(controller.warning_msg, "")

    def test_select_trained_cnn_mode_starts_background_load(self):
        """
        Khi chưa có cache, nút VS TRAINED CNN chỉ khởi động thread load nền.

        Test này đảm bảo event click không gọi get_trained_ai trực tiếp, vì đó
        là chỗ import TensorFlow/Keras và load file model gây đứng giao diện.
        """
        controller = self.make_controller_without_ui()
        controller.game_state = STATE_MENU
        controller.get_cnn_model_signature = lambda: 789
        controller.get_trained_ai = mock.Mock(side_effect=AssertionError("should run in background thread only"))

        class FakeThread:
            def __init__(self, target, name=None):
                self.target = target
                self.name = name
                self.daemon = False
                self.started = False

            def is_alive(self):
                return False

            def start(self):
                self.started = True

        with mock.patch("src.main.threading.Thread", FakeThread):
            controller.select_trained_cnn_mode()

        controller.get_trained_ai.assert_not_called()
        self.assertIsInstance(controller.cnn_load_thread, FakeThread)
        self.assertTrue(controller.cnn_load_thread.started)
        self.assertEqual(controller.game_state, STATE_MENU)
        self.assertEqual(controller.warning_msg, "Loading CNN model...")


@unittest.skipIf(importlib.util.find_spec("keras") is None, "keras is not installed")
class SupervisedValidationTest(unittest.TestCase):
    def setUp(self):
        """
        Tạo SupervisedAI cho từng test validation dữ liệu train.

        Việc khởi tạo trong setUp giúp mỗi test có một instance riêng, tránh
        trạng thái model hoặc đường dẫn bị chia sẻ ngoài ý muốn.
        """
        from ai.supervised import SupervisedAI

        self.ai = SupervisedAI(model_path='data/models/__test_missing_model__.h5')

    

    def test_rejects_training_step_on_occupied_cell(self):
        """
        Đảm bảo bước train bị loại nếu nước đi trỏ vào ô đã có quân.

        CNN chỉ nên học những action hợp lệ, vì vậy log có move sai trạng
        thái bàn cờ phải bị bỏ qua.
        """
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[7][7] = 1
        step = {
            "board": board,
            "move": [7, 7],
            "player": 1,
            "source": "human",
        }

        self.assertFalse(self.ai.is_trainable_step(step, winner=1))

    def test_accepts_valid_winning_training_step(self):
        """
        Chấp nhận bước hợp lệ của bên thắng từ nguồn dữ liệu huấn luyện.

        Test xác nhận is_trainable_step vẫn nhận dữ liệu human/minimax hợp lệ
        để không vô tình loại bỏ dữ liệu huấn luyện tốt.
        """
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[7][6] = 1
        board[6][6] = 2
        step = {
            "board": board,
            "move": [7, 7],
            "player": 1,
            "source": "human",
        }

        self.assertTrue(self.ai.is_trainable_step(step, winner=1))

    def test_rejects_losing_player_step(self):
        """
        Loại bỏ nước đi của bên thua khỏi dữ liệu train CNN.

        Pipeline supervised hiện chỉ học từ người thắng, nên bước của player
        khác winner phải trả về False.
        """
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        step = {
            "board": board,
            "move": [7, 7],
            "player": 2,
            "source": "human",
        }

        self.assertFalse(self.ai.is_trainable_step(step, winner=1))

    def test_rejects_self_generated_cnn_step(self):
        """
        Loại bỏ nước đi do chính CNN sinh ra khỏi tập dữ liệu tham chiếu.

        Điều này tránh vòng lặp tự học từ dự đoán của chính model supervised,
        vốn có thể khuếch đại lỗi.
        """
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        step = {
            "board": board,
            "move": [7, 7],
            "player": 1,
            "source": "cnn",
        }

        self.assertFalse(self.ai.is_trainable_step(step, winner=1))

    def test_normalizes_board_from_player_perspective(self):
        """
        Kiểm tra chuẩn hóa bàn cờ theo góc nhìn của player được train.

        Quân của player phải thành 1 và quân đối thủ thành 2 để một model có
        thể học chung cho cả hai phe.
        """
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[3][3] = 2
        board[3][4] = 1

        normalized = self.ai.normalize_board_for_player(board, player=2)

        self.assertEqual(normalized[3][3], 1)
        self.assertEqual(normalized[3][4], 2)

    def test_encode_board_channels_separates_cell_types(self):
        """
        Đảm bảo encoding CNN mới tách quân mình, quân đối thủ và ô trống.

        Việc dùng 3 kênh giúp model không hiểu nhầm mã quân 2 là giá trị lớn
        hơn về mặt số học so với mã quân 1.
        """
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        board[3][3] = 2
        board[3][4] = 1

        normalized = self.ai.normalize_board_for_player(board, player=2)
        encoded = self.ai.encode_board_channels(normalized)

        self.assertEqual(encoded.shape, (BOARD_SIZE, BOARD_SIZE, 3))
        self.assertEqual(encoded[3][3][0], 1)
        self.assertEqual(encoded[3][4][1], 1)
        self.assertEqual(encoded[0][0][2], 1)

    def test_prepare_board_input_keeps_legacy_scalar_model_compatible(self):
        """
        Đảm bảo model .h5 cũ vẫn nhận input 15x15 scalar.

        Khi load model đã train trước khi có encoding 3 kênh, inference không
        được đổi shape đầu vào một cách âm thầm.
        """
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.ai.model = type("LegacyModel", (), {"input_shape": (None, BOARD_SIZE, BOARD_SIZE)})()

        prepared = self.ai.prepare_board_input(board)

        self.assertEqual(prepared.shape, (BOARD_SIZE, BOARD_SIZE))

    def test_residual_cnn_contains_skip_connections(self):
        """
        Đảm bảo model mới dùng residual block thay cho CNN tuần tự quá nông.
        """
        model = self.ai.build_cnn_model()
        layer_types = [layer.__class__.__name__ for layer in model.layers]

        self.assertIn("Add", layer_types)
        self.assertIn("BatchNormalization", layer_types)
        self.assertEqual(model.output_shape[-1], BOARD_SIZE * BOARD_SIZE)

    def test_rejects_invalid_prediction_player(self):
        """
        Đảm bảo CNN không chuẩn hóa hoặc predict với mã player ngoài 1 và 2.

        Nếu player sai, get_best_move phải trả về None trước khi gọi model, còn
        normalize_board_for_player phải báo lỗi rõ ràng cho caller.
        """
        class FailingModel:
            def predict(self, *_args, **_kwargs):
                """
                Giả lập API predict của Keras model nhưng luôn báo lỗi.

                Test dùng fake này để đảm bảo get_best_move trả về None trước
                khi gọi model khi player không hợp lệ.
                """
                raise AssertionError("predict should not be called for invalid player")

            

        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.ai.model = FailingModel()

        self.assertIsNone(self.ai.get_best_move(board, player=0))
        with self.assertRaises(ValueError):
            self.ai.normalize_board_for_player(board, player=0)


def load_tests(loader, tests, pattern):
    """
    Nạp rõ ràng các TestCase chính của dự án.
    """
    suite = unittest.TestSuite()
    for case in (BoardLogicTest, MinimaxLogicTest, EvaluateScriptTest, GameControllerStateTest, SupervisedValidationTest):
        suite.addTests(loader.loadTestsFromTestCase(case))
    return suite


if __name__ == "__main__":
    unittest.main()
