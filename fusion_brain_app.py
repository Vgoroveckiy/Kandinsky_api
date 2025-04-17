import io
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from dotenv import load_dotenv

# Импортируем ваш существующий клиент
from fusionbrain_client import FusionBrainClient
from PIL import Image, ImageTk

# Загружаем переменные окружения
load_dotenv()


class FusionBrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FusionBrain Image Generator")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        # Инициализация клиента
        try:
            self.client = FusionBrainClient()
            self.models = self.client.get_models()
        except Exception as e:
            messagebox.showerror("Error", f"Ошибка инициализации API: {e}")
            self.models = []

        # Переменные для хранения параметров
        self.prompt_var = tk.StringVar(value="Красивый закат на морском побережье")
        self.neg_prompt_var = tk.StringVar(value="грязь, плохое качество, искажения")
        self.model_id_var = tk.IntVar(value=1)
        self.width_var = tk.IntVar(value=1024)
        self.height_var = tk.IntVar(value=1024)
        self.images_num_var = tk.IntVar(value=1)
        self.style_var = tk.StringVar()
        self.guidance_scale_var = tk.DoubleVar(value=7.0)
        self.seed_var = tk.StringVar()

        # Список для хранения сгенерированных изображений
        self.generated_images = []
        self.current_image_index = 0

        # Создание интерфейса
        self.create_widgets()

    def create_widgets(self):
        # Основной фрейм с двумя частями: параметры и изображение
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель с параметрами
        params_frame = ttk.LabelFrame(main_frame, text="Параметры генерации")
        main_frame.add(params_frame, weight=1)

        # Правая панель с изображением
        image_frame = ttk.LabelFrame(main_frame, text="Предпросмотр изображения")
        main_frame.add(image_frame, weight=2)

        # Настройка левой панели с параметрами
        self.setup_params_panel(params_frame)

        # Настройка правой панели с изображением
        self.setup_image_panel(image_frame)

    def setup_params_panel(self, parent):
        params_canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(
            parent, orient="vertical", command=params_canvas.yview
        )
        scrollable_frame = ttk.Frame(params_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: params_canvas.configure(scrollregion=params_canvas.bbox("all")),
        )

        params_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        params_canvas.configure(yscrollcommand=scrollbar.set)

        params_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Prompt
        ttk.Label(scrollable_frame, text="Prompt:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        prompt_entry = ttk.Entry(
            scrollable_frame, textvariable=self.prompt_var, width=40
        )
        prompt_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # Negative Prompt
        ttk.Label(scrollable_frame, text="Negative Prompt:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        neg_prompt_entry = ttk.Entry(
            scrollable_frame, textvariable=self.neg_prompt_var, width=40
        )
        neg_prompt_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Model Selection
        ttk.Label(scrollable_frame, text="Модель:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5
        )
        model_combo = ttk.Combobox(scrollable_frame, state="readonly")
        if self.models:
            model_combo["values"] = [
                f"{model['id']}: {model['name']}" for model in self.models
            ]
            model_combo.current(0)
        model_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        model_combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self.model_id_var.set(int(model_combo.get().split(":")[0])),
        )

        # Размеры
        size_frame = ttk.LabelFrame(scrollable_frame, text="Размеры изображения")
        size_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Ширина
        ttk.Label(size_frame, text="Ширина:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        width_combo = ttk.Combobox(
            size_frame, textvariable=self.width_var, state="readonly"
        )
        width_combo["values"] = (512, 768, 1024)
        width_combo.current(2)  # 1024 по умолчанию
        width_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # Высота
        ttk.Label(size_frame, text="Высота:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        height_combo = ttk.Combobox(
            size_frame, textvariable=self.height_var, state="readonly"
        )
        height_combo["values"] = (512, 768, 1024)
        height_combo.current(2)  # 1024 по умолчанию
        height_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # Кол-во изображений
        ttk.Label(scrollable_frame, text="Количество изображений:").grid(
            row=4, column=0, sticky="w", padx=5, pady=5
        )
        images_num_combo = ttk.Combobox(
            scrollable_frame, textvariable=self.images_num_var, state="readonly"
        )
        images_num_combo["values"] = (1, 2, 3, 4)
        images_num_combo.current(0)
        images_num_combo.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        # Стиль
        ttk.Label(scrollable_frame, text="Стиль (опционально):").grid(
            row=5, column=0, sticky="w", padx=5, pady=5
        )
        style_entry = ttk.Entry(scrollable_frame, textvariable=self.style_var, width=40)
        style_entry.grid(row=5, column=1, sticky="ew", padx=5, pady=5)

        # Guidance Scale
        ttk.Label(scrollable_frame, text="Guidance Scale (1-12):").grid(
            row=6, column=0, sticky="w", padx=5, pady=5
        )
        guidance_scale_slider = ttk.Scale(
            scrollable_frame,
            from_=1,
            to=12,
            orient=tk.HORIZONTAL,
            variable=self.guidance_scale_var,
        )
        guidance_scale_slider.grid(row=6, column=1, sticky="ew", padx=5, pady=5)
        guidance_scale_value = ttk.Label(
            scrollable_frame, textvariable=self.guidance_scale_var
        )
        guidance_scale_value.grid(row=6, column=2, padx=5, pady=5)

        # Seed
        ttk.Label(scrollable_frame, text="Seed (опционально):").grid(
            row=7, column=0, sticky="w", padx=5, pady=5
        )
        seed_entry = ttk.Entry(scrollable_frame, textvariable=self.seed_var, width=40)
        seed_entry.grid(row=7, column=1, sticky="ew", padx=5, pady=5)

        # Кнопка генерации
        generate_button = ttk.Button(
            scrollable_frame, text="Сгенерировать", command=self.start_generation
        )
        generate_button.grid(row=8, column=0, columnspan=2, pady=20)

    def setup_image_panel(self, parent):
        # Фрейм для отображения изображения
        self.image_display_frame = ttk.Frame(parent)
        self.image_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Метка для отображения изображения
        self.image_label = ttk.Label(self.image_display_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Фрейм с кнопками управления
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        # Кнопки переключения между изображениями
        self.prev_button = ttk.Button(
            controls_frame, text="←", command=self.show_prev_image, state=tk.DISABLED
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.image_counter_label = ttk.Label(controls_frame, text="0 / 0")
        self.image_counter_label.pack(side=tk.LEFT, padx=10)

        self.next_button = ttk.Button(
            controls_frame, text="→", command=self.show_next_image, state=tk.DISABLED
        )
        self.next_button.pack(side=tk.LEFT, padx=5)

        # Кнопка сохранения
        self.save_button = ttk.Button(
            controls_frame,
            text="Сохранить",
            command=self.save_current_image,
            state=tk.DISABLED,
        )
        self.save_button.pack(side=tk.RIGHT, padx=5)

    def start_generation(self):
        # Получаем параметры из полей формы
        params = {
            "prompt": self.prompt_var.get(),
            "negative_prompt": self.neg_prompt_var.get(),
            "model_id": self.model_id_var.get(),
            "width": self.width_var.get(),
            "height": self.height_var.get(),
            "images_num": self.images_num_var.get(),
            "guidance_scale": self.guidance_scale_var.get(),
        }

        # Добавляем опциональные параметры
        if self.style_var.get():
            params["style"] = self.style_var.get()

        if self.seed_var.get():
            try:
                params["seed"] = int(self.seed_var.get())
            except ValueError:
                messagebox.showerror("Ошибка", "Seed должен быть числом")
                return

        # Показываем индикатор загрузки
        self.image_label.configure(
            text="Генерация изображения... Пожалуйста, подождите."
        )

        # Запускаем генерацию в отдельном потоке
        threading.Thread(
            target=self.generate_image, args=(params,), daemon=True
        ).start()

    def generate_image(self, params):
        try:
            # Генерируем изображение
            self.generated_images = self.client.generate_image(**params)

            # Сбрасываем индекс текущего изображения
            self.current_image_index = 0

            # Обновляем интерфейс в главном потоке
            self.root.after(0, self.update_image_display)

        except Exception as e:
            # Обрабатываем ошибки
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Ошибка", f"Не удалось сгенерировать изображение: {e}"
                ),
            )
            self.root.after(
                0,
                lambda: self.image_label.configure(
                    text="Произошла ошибка при генерации."
                ),
            )

    def update_image_display(self):
        if not self.generated_images:
            return

        # Показываем текущее изображение
        self.display_image(self.generated_images[self.current_image_index])

        # Обновляем счетчик и состояние кнопок
        self.image_counter_label.configure(
            text=f"{self.current_image_index + 1} / {len(self.generated_images)}"
        )

        # Активируем/деактивируем кнопки навигации
        self.prev_button.configure(
            state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED
        )
        self.next_button.configure(
            state=(
                tk.NORMAL
                if self.current_image_index < len(self.generated_images) - 1
                else tk.DISABLED
            )
        )

        # Активируем кнопку сохранения
        self.save_button.configure(state=tk.NORMAL)

    def display_image(self, pil_image):
        # Получаем размеры области отображения
        display_width = self.image_display_frame.winfo_width()
        display_height = self.image_display_frame.winfo_height()

        # Если размеры еще не определены (при первом запуске)
        if display_width <= 1:
            display_width = 600
        if display_height <= 1:
            display_height = 600

        # Масштабируем изображение для отображения
        img_copy = pil_image.copy()
        img_copy.thumbnail((display_width, display_height), Image.LANCZOS)

        # Конвертируем в формат Tkinter
        tk_image = ImageTk.PhotoImage(img_copy)

        # Сохраняем ссылку (иначе сборщик мусора может удалить изображение)
        self.current_tk_image = tk_image

        # Обновляем метку с изображением
        self.image_label.configure(image=tk_image, text="")

    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image_display()

    def show_next_image(self):
        if self.current_image_index < len(self.generated_images) - 1:
            self.current_image_index += 1
            self.update_image_display()

    def save_current_image(self):
        if not self.generated_images or self.current_image_index >= len(
            self.generated_images
        ):
            return

        # Открываем диалог сохранения файла
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*"),
            ],
            title="Сохранить изображение как",
        )

        if file_path:
            try:
                # Сохраняем текущее изображение
                self.generated_images[self.current_image_index].save(file_path)
                messagebox.showinfo(
                    "Успех", f"Изображение успешно сохранено в {file_path}"
                )
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FusionBrainApp(root)
    root.mainloop()
