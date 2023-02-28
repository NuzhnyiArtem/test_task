import pickle
import cv2 as cv
import numpy as np
from tkinter.filedialog import askopenfilename
import time
from loguru import logger
from typing import List, Dict, Callable
from os import listdir


def method_decorator(func: Callable):
    """Декоратор логгирования"""
    def wrapper(self, *args, **kwargs) -> Dict[str, int]:
        logger.debug(f"Запускается {func.__name__}")
        start = time.time()
        res = func(self, *args, **kwargs)
        end = time.time()
        logger.debug(f"Завершение {func.__name__}")
        logger.debug(f'Время работы {end - start}')
        if isinstance(res, list) and len(res) > 0:
            logger.debug(f'Для {self.fig_key} Найдено {len(res)} совпадений')
        elif isinstance(res, list) and len(res) == 0:
            logger.debug(f'Для {self.fig_key} Совпадений не найдено')
        elif isinstance(res, dict):
            logger.debug(f'Результат: {res}')
        return res

    return wrapper


class Rules:

    def __init__(self, template: List[str],
                 threshold: float,
                 fig_key: str) -> None:

        self.template = template
        self.threshold = threshold
        self.fig_key = fig_key

    @method_decorator
    def find_matches(self, image) -> List[List[int]]:
        """Принимает экземпляр класса Rules
        и производит поиск совпадений по шаблонам
        возвращает список координат фигуры
        """
        try:
            coords: List = []
            for figure in self.template:
                img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                temp = cv.imread(figure + ".png", 0)
                w, h = temp.shape[::-1]
                res = cv.matchTemplate(img_gray, temp, cv.TM_CCOEFF_NORMED)
                yloc, xloc = np.where(res >= self.threshold)
                for (x, y) in zip(xloc, yloc):
                    coords.append([x, y, w, h])
                    coords.append([x, y, w, h])
            if len(coords) > 0:
                coords, weights = cv.groupRectangles(coords, 1, 0.2)
                return np.ndarray.tolist(coords)
            return coords

        except Exception as e:
            logger.exception(e)

    @method_decorator
    def save_pickle(self):
        with open(self.fig_key + ".pickle", 'wb') as f:
            pickle.dump(self, f)


class Paint:

    def __init__(self, image: np.ndarray,
                 result: List = [],
                 templates: List = []) -> None:

        self.image = image
        self.templates = templates
        self.result = result

    @method_decorator
    def run(self, res_dict: Dict = {}) -> Dict[str, int]:
        """Принимает список экземпляров класса Rules,
        вызывает метод find_matches.
        После чего рисует контуры прямоугольников,
        сохраняет изображение с контурами
        и возвращает словарь с координатами фигур
        """
        try:
            for template in self.templates:
                self.result = template.find_matches(self.image)
                res_dict[template.fig_key] = self.result
                for (x, y, w, h) in self.result:
                    cv.rectangle(self.image, (x, y), (x + w, y + h),
                                 (0, 0, 255), 2)
                cv.imwrite('output1.jpeg', self.image)
            return res_dict

        except Exception as e:
            logger.exception(e)

    @method_decorator
    def read_pickle(self) -> None:
        """Создает список экземпляров класса Rules
        из файлов и передает в метод run
        """
        try:
            files = listdir(mypath)
            pickles = filter(lambda x: x.endswith('.pickle'), files)
            for pic in pickles:
                with open(pic, 'rb') as f:
                    loads = pickle.load(f)
                    self.templates.append(loads)
            self.run()

        except PermissionError:
            logger.exception("Не найден файл")

        except FileNotFoundError:
            logger.exception("Системе не удается найти указанный путь")

        except Exception as e:
            logger.exception(e)


image = cv.imread(askopenfilename())

mypath = r""
p = Paint(image)
p.read_pickle()
