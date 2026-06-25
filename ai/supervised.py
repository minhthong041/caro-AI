import os
import json
import csv
import random
import numpy as np
import shutil

os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from keras.models import Model, load_model
from keras.layers import Add, Activation, BatchNormalization, Conv2D, Dense, Flatten, Input, Reshape
from keras.metrics import SparseTopKCategoricalAccuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from src.config import BOARD_SIZE

class SupervisedAI:
    TRAINABLE_SOURCES = {None, 'human', 'minimax'}
    VALID_PLAYERS = (1, 2)
    CHANNEL_COUNT = 3

    def __init__(self, model_path='data/models/caro_supervised.h5', load_existing=True):
        """
        Nạp hoặc chuẩn bị mô hình CNN dùng để dự đoán nước đi caro.

        model_path là đường dẫn tương đối từ thư mục gốc dự án. Nếu file
        model đã tồn tại, hàm sẽ load ngay để có thể chơi hoặc tiếp tục train;
        nếu chưa có hoặc load_existing=False, model được tạo khi gọi
        build_cnn_model/train_model.
        """
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', model_path))
        self.board_size = BOARD_SIZE
        self.model = None
        
        if load_existing and os.path.exists(self.model_path):
            self.model = load_model(self.model_path, compile=False)
            print("[Supervised] CNN model loaded successfully!")

    def is_valid_player(self, player):
        """
        Kiểm tra mã người chơi có thuộc hai phe hợp lệ của bàn cờ không.

        SupervisedAI chỉ chuẩn hóa và dự đoán cho player 1 hoặc 2. Nếu nhận
        giá trị khác, các hàm gọi bên ngoài có thể dừng sớm thay vì đưa dữ
        liệu sai vào model Keras.
        """
        try:
            return int(player) in self.VALID_PLAYERS
        except (TypeError, ValueError):
            return False

    def get_training_metrics(self):
        """
        Trả về bộ metric dùng khi train CNN.

        Bên cạnh accuracy top-1, top-3 accuracy hữu ích cho bài toán dự đoán
        nước đi vì nhiều ô có thể gần tương đương chiến thuật trong cùng một thế.
        """
        return ['accuracy', SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]

    def residual_block(self, x, filters):
        """
        Tạo một residual block 3x3 + 3x3 với skip-connection.

        Skip-connection giúp gradient đi qua mạng sâu ổn định hơn so với việc
        xếp nhiều Conv2D tuần tự, phù hợp hơn với bài toán nhận diện thế cờ.
        """
        shortcut = x
        x = Conv2D(filters, kernel_size=(3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size=(3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        if shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False)(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        return Activation('relu')(x)

    def build_cnn_model(self, input_channels=CHANNEL_COUNT, learning_rate=0.001):
        """
        Tạo Residual CNN nhận bàn cờ và xuất xác suất cho mọi ô.

        Mỗi ô đầu ra tương ứng một hành động đặt quân. Hàm chỉ xây dựng và
        compile model, chưa lưu ra ổ cứng và chưa huấn luyện dữ liệu.
        """
        if input_channels == self.CHANNEL_COUNT:
            inputs = Input(shape=(self.board_size, self.board_size, self.CHANNEL_COUNT))
            x = inputs
        else:
            inputs = Input(shape=(self.board_size, self.board_size))
            x = Reshape((self.board_size, self.board_size, 1))(inputs)

        x = Conv2D(64, kernel_size=(3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        for _ in range(3):
            x = self.residual_block(x, filters=64)

        policy = Conv2D(2, kernel_size=(1, 1), activation='relu', padding='same')(x)
        policy = Flatten()(policy)
        policy = Dense(256, activation='relu')(policy)
        outputs = Dense(self.board_size * self.board_size, activation='softmax')(policy)

        model = Model(inputs=inputs, outputs=outputs, name='caro_resnet_policy')
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=self.get_training_metrics(),
        )
        return model

    def compile_model_for_training(self, learning_rate=0.001):
        """
        Compile model khi thật sự train; predict không cần bước này.
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=self.get_training_metrics(),
        )

    def get_symmetries(self, board, r, c):
        """
        Tạo 8 biến thể đối xứng cho một mẫu huấn luyện.

        Caro giữ nguyên ý nghĩa chiến thuật khi xoay hoặc lật bàn cờ, nên
        augmentation này giúp tăng dữ liệu train mà không cần thêm log mới.
        Hàm trả về cặp (bàn cờ đã biến đổi, action index mới).
        """
        symmetries = []
        
        move_board = np.zeros((self.board_size, self.board_size))
        move_board[r][c] = 1

        for i in range(4):
            rotated_board = np.rot90(board, i)
            rotated_move_board = np.rot90(move_board, i)

            new_r, new_c = np.where(rotated_move_board == 1)
            new_move_1d = new_r[0] * self.board_size + new_c[0]
            symmetries.append((rotated_board, new_move_1d))

            flipped_board = np.fliplr(rotated_board)
            flipped_move_board = np.fliplr(rotated_move_board)

            new_r_f, new_c_f = np.where(flipped_move_board == 1)
            new_move_1d_f = new_r_f[0] * self.board_size + new_c_f[0]
            symmetries.append((flipped_board, new_move_1d_f))
            
        return symmetries
    
    def normalize_board_for_player(self, board, player):
        """
        Chuẩn hóa bàn cờ theo góc nhìn của người chơi cần học.

        Quân của player luôn được mã hóa là 1, quân đối thủ là 2. Cách này
        giúp cùng một model học được mẫu chiến thuật cho cả hai phe.
        """
        if not self.is_valid_player(player):
            raise ValueError("player must be 1 or 2")

        board_np = np.array(board)
        player = int(player)
        opponent = 2 if player == 1 else 1

        normalized = np.zeros_like(board_np)
        normalized[board_np == player] = 1
        normalized[board_np == opponent] = 2
        return normalized

    def encode_board_channels(self, normalized_board):
        """
        Mã hóa bàn cờ thành 3 kênh nhị phân: quân mình, quân đối thủ và ô trống.

        Encoding nhiều kênh giúp CNN nhìn từng loại ô như những đặc trưng riêng,
        thay vì xem quân đối thủ là một giá trị số lớn hơn quân của mình.
        """
        board_np = np.array(normalized_board)
        encoded = np.zeros((self.board_size, self.board_size, self.CHANNEL_COUNT), dtype=np.float32)
        encoded[:, :, 0] = (board_np == 1)
        encoded[:, :, 1] = (board_np == 2)
        encoded[:, :, 2] = (board_np == 0)
        return encoded

    def get_model_input_shape(self):
        """
        Lấy input shape của model Keras hiện tại nếu có.

        Keras đôi khi trả input_shape dạng list với model nhiều input; dự án chỉ
        dùng một input nên lấy phần tử đầu để các nhánh xử lý thống nhất.
        """
        shape = getattr(self.model, 'input_shape', None)
        if isinstance(shape, list) and shape:
            shape = shape[0]
        return shape

    def uses_channel_encoding(self):
        """
        Kiểm tra model hiện tại dùng input 3 kênh hay định dạng scalar cũ.
        """
        shape = self.get_model_input_shape()
        return isinstance(shape, tuple) and len(shape) == 4 and shape[-1] == self.CHANNEL_COUNT

    def prepare_board_input(self, normalized_board):
        """
        Chuẩn bị input đúng shape cho model hiện tại.

        Model mới dùng 3 kênh, còn các model .h5 cũ của dự án vẫn nhận ma trận
        15x15 scalar. Nhánh này giữ khả năng chơi lại với model đã train trước.
        """
        if self.model is not None and not self.uses_channel_encoding():
            return np.array(normalized_board, dtype=np.float32)
        return self.encode_board_channels(normalized_board)

    def is_trainable_step(self, step, winner):
        """
        Xác định một bước trong log có nên dùng để train CNN hay không.

        Chỉ lấy nước đi của bên thắng và từ các nguồn dữ liệu tham chiếu,
        nhằm tránh đưa vào các nước do model học sâu tự sinh ra.
        """
        if not isinstance(step, dict):
            return False
            
        source = step.get('source')
        if step.get('player') != winner or source not in self.TRAINABLE_SOURCES:
            return False
            
        board = step.get('board')
        move = step.get('move')
        if not isinstance(board, list) or len(board) != self.board_size:
            return False
        if any(not isinstance(row, list) or len(row) != self.board_size for row in board):
            return False
        if not isinstance(move, (list, tuple)) or len(move) != 2:
            return False
            
        try:
            r, c = int(move[0]), int(move[1])
        except (TypeError, ValueError):
            return False
            
        if not (0 <= r < self.board_size and 0 <= c < self.board_size):
            return False
        if step.get('player') not in (1, 2):
            return False
        if any(cell not in (0, 1, 2) for row in board for cell in row):
            return False
        if board[r][c] != 0:
            return False
            
        return True
    
    def analyze_training_file(self, logs_path, filename):
        """
        Phân tích một file log và trả về số sample có thể dùng để train.

        valid_steps là số nước gốc trước augmentation; samples là số mẫu sau
        augmentation đối xứng. skip_reason giúp biết vì sao file không được
        đưa vào lần train hiện tại.
        """
        file_path = os.path.join(logs_path, filename)
        result = {
            "filename": filename,
            "valid_steps": 0,
            "samples": 0,
            "skip_reason": None
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            result["skip_reason"] = f"cannot read JSON ({e})"
            return result
            
        winner = data.get('winner')
        if winner not in (1, 2):
            result["skip_reason"] = "game has no winner"
            return result
            
        steps = data.get('history', [])
        if not isinstance(steps, list):
            result["skip_reason"] = "history is missing or invalid"
            return result
            
        valid_steps = sum(1 for step in steps if self.is_trainable_step(step, winner))
        if valid_steps == 0:
            result["skip_reason"] = "no winning human/minimax moves"
            return result
            
        result["valid_steps"] = valid_steps
        result["samples"] = valid_steps * 8
        return result
    
    def build_training_report(self, logs_path, files):
        """
        Tạo báo cáo scan toàn bộ log trước khi train.

        Report này giúp xác nhận lần train đang dùng tất cả file JSON hợp lệ
        trong human_logs, thay vì âm thầm bỏ qua khiến dễ hiểu nhầm là chỉ
        học từ một file.
        """
        train_files = []
        skipped_files = []
        total_samples = 0
        total_steps = 0
        
        for filename in sorted(files):
            report = self.analyze_training_file(logs_path, filename)
            if report["samples"] > 0:
                train_files.append(filename)
                total_samples += report["samples"]
                total_steps += report["valid_steps"]
            else:
                skipped_files.append(report)
                
        return train_files, skipped_files, total_steps, total_samples

    def split_training_files(self, files, validation_split=0.2, seed=0, shuffle=True):
        """
        Tách file log thành train/validation theo cách quyết định được.

        Với ít nhất hai file, một phần dữ liệu được giữ lại để đo validation.
        Khi shuffle=True, thứ tự file được trộn bằng seed cố định để validation
        không phụ thuộc vào tên file/timestamp nhưng vẫn lặp lại được.
        """
        files = sorted(files)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(files)
        if validation_split <= 0 or len(files) < 2:
            return files, []

        validation_count = int(round(len(files) * validation_split))
        validation_count = max(1, min(validation_count, len(files) - 1))
        return files[:-validation_count], files[-validation_count:]

    def count_samples_for_files(self, logs_path, files):
        """
        Đếm số nước gốc và số sample sau augmentation cho một nhóm file.
        """
        total_steps = 0
        total_samples = 0
        for filename in files:
            report = self.analyze_training_file(logs_path, filename)
            total_steps += report["valid_steps"]
            total_samples += report["samples"]
        return total_steps, total_samples

    def build_dataset(self, logs_path, files):
        """
        Nạp một tập dữ liệu hữu hạn vào RAM, dùng cho validation.

        Train vẫn dùng generator để tiết kiệm bộ nhớ; validation thường nhỏ hơn
        nên nạp trực tiếp giúp Keras tính metric chính xác sau mỗi epoch.
        """
        X, y = [], []
        for filename in files:
            file_path = os.path.join(logs_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue

            winner = data.get('winner')
            if winner not in (1, 2):
                continue

            steps = data.get('history', [])
            if not isinstance(steps, list):
                continue

            for step in steps:
                if self.is_trainable_step(step, winner):
                    board_state = self.normalize_board_for_player(step['board'], step['player'])
                    r, c = int(step['move'][0]), int(step['move'][1])
                    for aug_board, aug_move in self.get_symmetries(board_state, r, c):
                        X.append(self.prepare_board_input(aug_board))
                        y.append(aug_move)

        if not X:
            return None
        return np.array(X), np.array(y)

    def save_training_history(self, history, report):
        """
        Lưu loss/accuracy/top-k sau khi train để đối chiếu kết quả huấn luyện.
        """
        history_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'training_history'))
        os.makedirs(history_path, exist_ok=True)

        history_data = {
            metric: [float(value) for value in values]
            for metric, values in history.history.items()
        }
        payload = {
            "report": report,
            "history": history_data,
        }

        json_path = os.path.join(history_path, 'cnn_training_history.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

        csv_path = os.path.join(history_path, 'cnn_training_history.csv')
        metrics = list(history_data.keys())
        epoch_count = max((len(values) for values in history_data.values()), default=0)
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'] + metrics)
            for epoch in range(epoch_count):
                writer.writerow([
                    epoch + 1,
                    *[
                        history_data[metric][epoch] if epoch < len(history_data[metric]) else ''
                        for metric in metrics
                    ],
                ])

        plot_path = None
        try:
            import matplotlib.pyplot as plt

            plot_path = os.path.join(history_path, 'cnn_training_history.png')
            plt.figure(figsize=(9, 5))
            for metric, values in history_data.items():
                plt.plot(range(1, len(values) + 1), values, marker='o', label=metric)
            plt.xlabel('Epoch')
            plt.ylabel('Metric value')
            plt.title('Caro CNN training history')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except Exception as e:
            print(f"Training plot was not generated: {e}")

        print(f"Saved training history to {json_path}")
        print(f"Saved training metrics table to {csv_path}")
        if plot_path:
            print(f"Saved training plot to {plot_path}")
        return json_path, csv_path, plot_path

    def build_training_callbacks(self, monitor, checkpoint_path=None):
        """
        Tạo callback giúp train ổn định hơn khi dữ liệu ít.

        EarlyStopping trả model về epoch tốt nhất, ReduceLROnPlateau giảm learning
        rate khi metric không cải thiện, còn checkpoint lưu bản tốt nhất ra file
        riêng để không phụ thuộc duy nhất vào model cuối cùng.
        """
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=4,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=1,
            ),
        ]
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor=monitor,
                    save_best_only=True,
                    verbose=1,
                )
            )
        return callbacks
    
    def get_archive_destination(self, archive_path, filename):
        """
        Tạo đường dẫn archive không ghi đè file cũ nếu trùng tên.

        Khi nhiều file log có cùng tên hoặc đã từng được train trước đó, hàm
        thêm hậu tố số tăng dần để giữ lại toàn bộ lịch sử thay vì overwrite
        dữ liệu đã archive.
        """
        base, ext = os.path.splitext(filename)
        destination = os.path.join(archive_path, filename)
        counter = 1
        
        while os.path.exists(destination):
            destination = os.path.join(archive_path, f"{base}_{counter}{ext}")
            counter += 1
            
        return destination
    
    def data_generator(self, logs_path, files=None, batch_size=32, shuffle=True, seed=0):
        """
        Sinh batch dữ liệu train CNN theo kiểu đọc cuốn chiếu từ file log.

        Generator không nạp toàn bộ log vào RAM cùng lúc. Mỗi batch gồm bàn
        cờ đã chuẩn hóa và action index của nước đi tương ứng.
        """
        X_batch, y_batch = [], []
        selected_files = list(files) if files is not None else None
        rng = random.Random(seed)
        
        # Keras fit() yêu cầu generator lặp cho tới khi đủ steps_per_epoch.
        while True:
            files_to_read = selected_files
            if files_to_read is None:
                files_to_read = [f for f in os.listdir(logs_path) if f.endswith(".json")]
            else:
                files_to_read = list(files_to_read)
            if not files_to_read:
                return
            if shuffle:
                rng.shuffle(files_to_read)
                
            yielded_any = False
                
            for filename in files_to_read:
                file_path = os.path.join(logs_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
                    continue
                    
                winner = data.get('winner')
                if winner not in (1, 2):
                    continue
                
                steps = data.get('history', [])
                if not isinstance(steps, list):
                    continue

                steps = [step for step in steps if self.is_trainable_step(step, winner)]
                if shuffle:
                    rng.shuffle(steps)
                    
                for step in steps:
                    board_state = self.normalize_board_for_player(step['board'], step['player'])
                    r, c = int(step['move'][0]), int(step['move'][1])
                    symmetries = self.get_symmetries(board_state, r, c)
                    if shuffle:
                        rng.shuffle(symmetries)
                    
                    for aug_board, aug_move in symmetries:
                        X_batch.append(self.prepare_board_input(aug_board))
                        y_batch.append(aug_move)
                        
                        if len(X_batch) >= batch_size:
                            yielded_any = True
                            yield np.array(X_batch), np.array(y_batch)
                            X_batch, y_batch = [], []
                    
            if X_batch:
                yielded_any = True
                yield np.array(X_batch), np.array(y_batch)
                X_batch, y_batch = [], []
                
            if not yielded_any:
                return

    def train_model(
        self,
        logs_dir='data/human_logs',
        validation_split=0.2,
        rebuild_model=False,
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
        seed=0,
        shuffle=True,
        archive=True,
        checkpoint=True,
    ):
        """
        Huấn luyện CNN từ các file log và lưu model sau khi train xong.

        Hàm tự tạo model mới nếu chưa load được model cũ, bỏ qua dữ liệu
        không hợp lệ, sau đó chuyển log đã xử lý sang thư mục archive để lần
        train sau không học lặp lại cùng dữ liệu.
        """
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0 <= validation_split < 1:
            raise ValueError("validation_split must be in the range [0, 1)")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        logs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', logs_dir))
        if not os.path.exists(logs_path): return
        
        all_files = sorted(f for f in os.listdir(logs_path) if f.endswith('.json'))
        if not all_files:
            print(f"No JSON log files found in {logs_path}.")
            return
        
        files_to_train, skipped_files, total_steps, total_samples = self.build_training_report(logs_path, all_files)
        print(f"Found {len(all_files)} JSON log files in {logs_dir}.")
        
        if total_samples == 0:
            print("No valid winning human/minimax samples found for CNN training.")
            for report in skipped_files:
                print(f"Skipped {report['filename']}: {report['skip_reason']}")
            return
        if skipped_files:
            print(f"Skipped {len(skipped_files)} log files with no usable training samples.")
            for report in skipped_files:
                print(f"Skipped {report['filename']}: {report['skip_reason']}")

        train_files, validation_files = self.split_training_files(
            files_to_train,
            validation_split=validation_split,
            seed=seed,
            shuffle=shuffle,
        )
        train_steps, train_samples = self.count_samples_for_files(logs_path, train_files)
        validation_steps, validation_samples = self.count_samples_for_files(logs_path, validation_files)

        print(f"Processing {len(files_to_train)} usable log files, {total_steps} moves, and {total_samples} augmented samples.")
        print(f"Training split: {len(train_files)} files, {train_steps} moves, {train_samples} samples.")
        if validation_files:
            print(f"Validation split: {len(validation_files)} files, {validation_steps} moves, {validation_samples} samples.")

        for filename in files_to_train:
            print(f"Using log: {filename}")

        steps_per_epoch = max(1, (train_samples + batch_size - 1) // batch_size)
        if rebuild_model or self.model is None:
            self.model = self.build_cnn_model(input_channels=self.CHANNEL_COUNT, learning_rate=learning_rate)
        else:
            self.compile_model_for_training(learning_rate=learning_rate)

        validation_data = self.build_dataset(logs_path, validation_files) if validation_files else None
        monitor = 'val_loss' if validation_data is not None else 'loss'
        checkpoint_path = None
        if checkpoint:
            checkpoint_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__),
                '..',
                'data',
                'models',
                'caro_supervised_best.keras',
            ))
        history = self.model.fit(
            self.data_generator(
                logs_path,
                files=train_files,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
            ),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=self.build_training_callbacks(monitor, checkpoint_path),
        )

        self.save_training_history(history, {
            "logs_dir": logs_dir,
            "all_files": len(all_files),
            "usable_files": len(files_to_train),
            "train_files": len(train_files),
            "validation_files": len(validation_files),
            "train_steps": train_steps,
            "validation_steps": validation_steps,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "shuffle": shuffle,
            "archive": archive,
            "best_checkpoint": checkpoint_path,
            "input_shape": str(self.get_model_input_shape()),
            "rebuild_model": rebuild_model,
        })
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        
        # Di chuyển đúng các file đã thật sự đóng góp sample sang archive.
        archive_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'logs_archive'))
        if os.path.abspath(logs_path) == os.path.abspath(archive_path):
            archive = False
            print("Archive skipped because logs_dir already points to data/logs_archive.")

        if archive:
            os.makedirs(archive_path, exist_ok=True)
            archived = 0
            for filename in files_to_train:
                file_path = os.path.join(logs_path, filename)
                if not os.path.exists(file_path):
                    continue
                shutil.move(file_path, self.get_archive_destination(archive_path, filename))
                archived += 1
            print(f"Archived {archived} trained log files.")
        else:
            print("Archive disabled; trained log files were left in place.")

    def get_best_move(self, grid, player=2):
        """
        Dự đoán nước đi tốt nhất cho player trên bàn cờ hiện tại.

        Model trả về xác suất cho toàn bộ ô trên bàn, nhưng hàm chỉ xét những ô
        còn trống để không bao giờ chọn một nước đi đã bị chiếm.
        """
        if self.model is None or not self.is_valid_player(player):
            return None
        normalized_board = self.normalize_board_for_player(grid, player)
        board_np = np.expand_dims(self.prepare_board_input(normalized_board), axis=0)
        predictions = self.model.predict(board_np, verbose=0)[0]
        
        valid_moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if grid[r][c] == 0]
        best_move, max_prob = None, -1
        
        for (r, c) in valid_moves:
            idx = r * self.board_size + c
            if predictions[idx] > max_prob:
                max_prob = predictions[idx]
                best_move = (r, c)
                
        return best_move
