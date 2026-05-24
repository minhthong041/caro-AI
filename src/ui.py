import pygame
import os

from src.config import BOARD_SIZE, CELL_SIZE

class GameUI:
    def __init__(self, board_size=BOARD_SIZE):
        """
        Khởi tạo toàn bộ tài nguyên giao diện Pygame cho game caro.

        Hàm load ảnh nền, ảnh quân cờ, font chữ, thiết lập kích thước cửa sổ
        và chuẩn bị các vùng Rect dùng để bắt click ở từng màn hình.
        """
        pygame.init()
        self.board_size = board_size

        # Thông số bố cục chính.
        self.GRID_OFFSET_X = 50
        self.GRID_OFFSET_Y = 50           
        self.CELL_SIZE = CELL_SIZE               
        self.GRID_COLOR = (120, 120, 120) 
        
        img_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images')
        font_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'fonts')

        # Kích thước cửa sổ theo ảnh nền chính.
        self.bg = pygame.image.load(os.path.join(img_dir, 'bg.png'))
        self.width = self.bg.get_width()
        self.height = self.bg.get_height()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Caro AI")
        
        # Tải và scale các ảnh nền theo kích thước cửa sổ.
        self.start_bg = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, 'start_bg.png')), (self.width, self.height))
        self.win_bg_custom = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, 'win_bg.png')), (self.width, self.height))
        self.lose_bg_custom = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, 'lose_bg.png')), (self.width, self.height))
        self.whofirst_bg = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, 'whofirst_bg.png')), (self.width, self.height))
        self.draw_bg_custom = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, 'draw_bg.png')), (self.width, self.height))
        
        # Tải tài nguyên quân cờ và nút bấm.
        self.piece_1 = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, 'piece_1.png')), (self.CELL_SIZE, self.CELL_SIZE))
        self.piece_2 = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, 'piece_2.png')), (self.CELL_SIZE, self.CELL_SIZE))
        self.btn_img = pygame.transform.scale(pygame.image.load(os.path.join(img_dir, 'button_bg.png')), (220, 60))
        btn_w, btn_h = self.btn_img.get_width(), self.btn_img.get_height()
        
        try:
            self.font = pygame.font.Font(os.path.join(font_dir, 'Minecraft.ttf'), 16)
            self.small_font = pygame.font.Font(os.path.join(font_dir, 'Minecraft.ttf'), 12)
        except (FileNotFoundError, pygame.error):
            self.font = pygame.font.SysFont('Arial', 16, bold=True)
            self.small_font = pygame.font.SysFont('Arial', 12, bold=True)
            
        self.TEXT_COLOR = (255, 255, 255)
        
        # Lớp phủ đánh dấu nước cờ cuối cùng.
        self.highlight_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE))
        self.highlight_surface.set_alpha(80) 
        self.highlight_surface.fill((0, 0, 0)) 
        
        center_x = self.width // 2 - btn_w // 2

        # Vùng bấm ở menu chính.
        self.rect_btn_minimax = pygame.Rect(center_x, 200, btn_w, btn_h)
        self.rect_btn_trained = pygame.Rect(center_x, 280, btn_w, btn_h)
        self.rect_btn_pvp = pygame.Rect(center_x, 360, btn_w, btn_h)
        
        # Vùng bấm ở màn hình chọn người đánh trước.
        self.rect_btn_p1_first = pygame.Rect(center_x, 300, btn_w, btn_h)
        self.rect_btn_p2_first = pygame.Rect(center_x, 380, btn_w, btn_h)
        self.rect_btn_back = pygame.Rect(center_x, 460, btn_w, btn_h)
        
        # Vùng bấm điều khiển trong ván chơi.
        panel_x = self.width - 250
        self.rect_replay = pygame.Rect(panel_x, self.height - 280, btn_w, btn_h)
        self.rect_menu = pygame.Rect(panel_x, self.height - 200, btn_w, btn_h)
        self.rect_exit = pygame.Rect(panel_x, self.height - 120, btn_w, btn_h)

        # Vùng bấm ở màn hình kết thúc.
        self.rect_replay_end = pygame.Rect(center_x, 350, btn_w, btn_h)
        self.rect_menu_end = pygame.Rect(center_x, 430, btn_w, btn_h)
        self.rect_exit_end = pygame.Rect(center_x, 510, btn_w, btn_h)

    def draw_text_centered(self, text, font, color, surface, rect):
        """
        Vẽ một chuỗi text nằm chính giữa vùng Rect cho trước.

        Hàm gom logic render text và canh tâm vào một chỗ để các màn hình
        menu, gameplay và kết thúc dùng chung cùng cách căn chỉnh.
        """
        text_obj = font.render(text, True, color)
        text_rect = text_obj.get_rect(center=rect.center)
        surface.blit(text_obj, text_rect)

    def draw_start_menu(self, warning_msg=""):
        """
        Vẽ màn hình menu chính và các nút chọn chế độ chơi.

        warning_msg được hiển thị ở cuối màn hình khi cần báo lỗi hoặc nhắc
        người chơi về trạng thái model.
        """
        self.screen.blit(self.start_bg, (0, 0))
        buttons = [
            (self.rect_btn_minimax, "VS MINIMAX"),
            (self.rect_btn_trained, "VS TRAINED CNN"),
            (self.rect_btn_pvp, "PvP TRAINING")
        ]
        for rect, txt in buttons:
            self.screen.blit(self.btn_img, rect)
            self.draw_text_centered(txt, self.font, self.TEXT_COLOR, self.screen, rect)
            
        if warning_msg:
            warning_rect = pygame.Rect(0, self.height - 80, self.width, 50)
            self.draw_text_centered(warning_msg, self.font, (255, 50, 50), self.screen, warning_rect)
        pygame.display.update()

    def draw_who_first_screen(self):
        """
        Vẽ màn hình chọn phe đi trước.

        Người chơi có thể chọn Angel (P1), Devil (P2) hoặc quay lại màn hình
        trước đó; controller sẽ reset ván sau khi chọn.
        """
        self.screen.blit(self.whofirst_bg, (0, 0))
        self.screen.blit(self.btn_img, self.rect_btn_p1_first)
        self.draw_text_centered("ANGEL (P1) FIRST", self.font, self.TEXT_COLOR, self.screen, self.rect_btn_p1_first)
        self.screen.blit(self.btn_img, self.rect_btn_p2_first)
        self.draw_text_centered("DEVIL (P2) FIRST", self.font, self.TEXT_COLOR, self.screen, self.rect_btn_p2_first)
        self.screen.blit(self.btn_img, self.rect_btn_back)
        self.draw_text_centered("BACK", self.font, self.TEXT_COLOR, self.screen, self.rect_btn_back)
        pygame.display.update()

    def draw_game_play(self, grid, current_player, mode_name, warning_msg="", last_move=None):
        """
        Vẽ bàn cờ, quân cờ, nước mới nhất và cụm điều khiển khi đang chơi.

        grid là ma trận bàn cờ hiện tại, current_player quyết định text lượt
        đi, mode_name hiển thị chế độ chơi và last_move dùng để highlight ô
        vừa được đánh.
        """
        self.screen.blit(self.bg, (0, 0))
        
        end_x = self.GRID_OFFSET_X + self.board_size * self.CELL_SIZE
        end_y = self.GRID_OFFSET_Y + self.board_size * self.CELL_SIZE
        
        # Vẽ lưới caro
        for i in range(self.board_size + 1):
            pygame.draw.line(self.screen, self.GRID_COLOR, (self.GRID_OFFSET_X + i*self.CELL_SIZE, self.GRID_OFFSET_Y), (self.GRID_OFFSET_X + i*self.CELL_SIZE, end_y))
            pygame.draw.line(self.screen, self.GRID_COLOR, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i*self.CELL_SIZE), (end_x, self.GRID_OFFSET_Y + i*self.CELL_SIZE))
        
        # Vẽ hệ tọa độ (số đếm) ở lề bàn cờ
        for i in range(self.board_size):
            txt_col = self.small_font.render(str(i), True, (0, 0, 0))
            self.screen.blit(txt_col, (self.GRID_OFFSET_X + i*self.CELL_SIZE + 15, self.GRID_OFFSET_Y - 20))
            txt_row = self.small_font.render(str(i), True, (0, 0, 0))
            self.screen.blit(txt_row, (self.GRID_OFFSET_X - 25, self.GRID_OFFSET_Y + i*self.CELL_SIZE + 15))
        
        # Vẽ quân cờ và đánh dấu nước đi gần nhất.
        for r in range(self.board_size):
            for c in range(self.board_size):
                pos = (self.GRID_OFFSET_X + c*self.CELL_SIZE, self.GRID_OFFSET_Y + r*self.CELL_SIZE)
                if grid[r][c] == 1: self.screen.blit(self.piece_1, pos)
                elif grid[r][c] == 2: self.screen.blit(self.piece_2, pos)
                
                if last_move and last_move == (r, c):
                    self.screen.blit(self.highlight_surface, pos)
                    
        # Vẽ cụm 3 nút điều khiển ở lề phải
        self.screen.blit(self.btn_img, self.rect_replay)
        self.draw_text_centered("REPLAY", self.font, self.TEXT_COLOR, self.screen, self.rect_replay)
        
        self.screen.blit(self.btn_img, self.rect_menu)
        self.draw_text_centered("MENU", self.font, self.TEXT_COLOR, self.screen, self.rect_menu)
        
        self.screen.blit(self.btn_img, self.rect_exit)
        self.draw_text_centered("BACK", self.font, self.TEXT_COLOR, self.screen, self.rect_exit)

        # Hiển thị chế độ chơi và lượt hiện tại ở khu vực điều khiển.
        self.draw_text_centered(f"Mode: {mode_name}", self.font, (0, 0, 0), self.screen, pygame.Rect(self.width - 250, self.height - 400, 220, 50))
        turn_text = "Turn: Angel (P1)" if current_player == 1 else "Turn: Devil (P2)"
        turn_color = (0, 150, 0) if current_player == 1 else (200, 0, 0) 
        self.draw_text_centered(turn_text, self.font, turn_color, self.screen, pygame.Rect(self.width - 250, self.height - 350, 220, 50))
            
        if warning_msg:
            warning_rect = pygame.Rect(self.GRID_OFFSET_X, end_y + 10, self.board_size * self.CELL_SIZE, 40)
            self.draw_text_centered(warning_msg, self.font, (255, 50, 50), self.screen, warning_rect)
            
        pygame.display.update()

    def draw_end_screen(self, winner_player, mode_name):
        """
        Vẽ màn hình kết quả sau khi ván đấu kết thúc.

        Background thay đổi theo kết quả thắng, thua hoặc hòa; các nút cuối
        ván cho phép chơi lại, về menu hoặc thoát game.
        """
        if winner_player == 0:
            self.screen.blit(self.draw_bg_custom, (0, 0)) 
        elif mode_name == "PvP Collect Data":
            self.screen.blit(self.win_bg_custom, (0, 0))  
        else:
            if winner_player == 1: self.screen.blit(self.win_bg_custom, (0, 0))
            else: self.screen.blit(self.lose_bg_custom, (0, 0))
            
        self.screen.blit(self.btn_img, self.rect_replay_end)
        self.draw_text_centered("REPLAY", self.font, self.TEXT_COLOR, self.screen, self.rect_replay_end)
        self.screen.blit(self.btn_img, self.rect_menu_end)
        self.draw_text_centered("MENU", self.font, self.TEXT_COLOR, self.screen, self.rect_menu_end)
        self.screen.blit(self.btn_img, self.rect_exit_end)
        self.draw_text_centered("EXIT", self.font, self.TEXT_COLOR, self.screen, self.rect_exit_end)
        pygame.display.update()
