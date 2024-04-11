import math
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox, kstest, shapiro, yeojohnson


def feature_engineering(data: pd.DataFrame, threshold: int = 25) -> pd.DataFrame:
    """
    Эта функция извлекает второе слово из столбца «заголовок» входного DataFrame.

    Параметры:
    data (pd.DataFrame): входной DataFrame, содержащий данные со столбцом «заголовок».

    Возврат:
    pd.DataFrame: DataFrame с дополнительным столбцом «бренды», содержащим второе слово из столбца «название».

    Поднимает:
    Исключение: если во время выполнения функции возникает ошибка.

    Функция применяет лямбда-функцию к каждой строке столбца «заголовок».
    Лямбда-функция извлекает второе слово из заголовка, используя шаблон регулярного выражения.
    Если второе слово существует, оно сохраняется в столбце «бренды» DataFrame.
    """

    def extract_second_word(text, word_number):
        """
        Эта функция извлекает N-ое слово из заданной текстовой строки.

        Параметры:
        text (str):  текстовую строку.

        Возврат:
        str: N-е слово входной текстовой строки, если оно существует. В противном случае возвращается Нет.

        Функция использует шаблон регулярного выражения для поиска всех слов во входной текстовой строке.
        Если есть хотя бы два слова, возвращается второе слово. В противном случае возвращается None.
        """

        # Паттерн для поиска слов в строке
        # word_pattern = r"\b[\w-]+\b"
        word_pattern = r"\b\w+(?:[-.]\w+)*\b"

        # Находим все слова в строке
        words = re.findall(word_pattern, text)

        # Если второе слово существует, возвращаем его
        if len(words) >= 2:
            return words[word_number]
        else:
            return text

    try:
        # бренд ноутбука
        data["brands"] = data["title"].apply(extract_second_word, args=(1,))

        # доставщик
        data["manufacturer"] = data["merchant_name"].apply(extract_second_word, args=(0,))

        # время погашения кредит
        data["loan_closure"] = data.apply(
            lambda row: math.ceil(row["price"] / row["credit"]) if row["credit"] != 0 else 0, axis=1
        )

        # Удаляем строки, у которых бренд ноутбука является малочисленным
        brand_counts = data["brands"].value_counts()
        manufacturer_counts = data["manufacturer"].value_counts()

        # Фильтруем бренды по порогу
        filtered_brands = brand_counts[brand_counts >= threshold].index
        filtered_manufacturer = manufacturer_counts[manufacturer_counts >= 4].index

        # Создаем новый DataFrame, содержащий только строки с брендами, встречающимися не менее threshold раз
        data = data[data["brands"].isin(filtered_brands)]
        data = data[data["manufacturer"].isin(filtered_manufacturer)]

        return data

    except Exception as e:
        print(e)
        return data


def missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Эта функция обрабатывает пропущенные значения в данном DataFrame.

    Параметры:
    data (pd.DataFrame): входной DataFrame, содержащий данные с пропущенными значениями.

    Возврат:
    pd.DataFrame: обработанный DataFrame с отсутствующими значениями.

    Возникает:
    Исключение: если во время выполнения функции возникает ошибка.

    Функция перебирает каждый столбец в DataFrame. В зависимости от имени столбца применяются разные методы для обработки пропущенных значений:
    – Если имя столбца — «бонус», «кредит» или «обзор», недостающие значения заполняются нулями.
    -Если имя столбца — «цена», недостающие значения заполняют средним значением существующих значений в столбце.
    -Для всех остальных столбцов он заполняет пропущенные значения наиболее частым значением (режимом) в столбце.

    Если во время выполнения функции возникает ошибка, она перехватывает исключение и печатает сообщение об ошибке.
    """

    try:
        for column in data.columns:
            if column is ["bonus", "credit", "review"]:
                data[column] = data.fillna(0)

            elif column == "price":
                data[column] = data[column].fillna(data[column].mean())

            else:
                data[column] = data[column].fillna(data[column].mode()[0])

        return data

    except Exception as e:
        print(e)
        return data


#### =========================================== Стат тесты для преобразованных данных ===============================


# Преобразование Йео-Джонса
def yeo_johnson_transform(data):
    # Применяем преобразование Йео-Джонсона
    transformed_data, lambda_value = yeojohnson(data)
    return transformed_data, lambda_value


# Преобразование Бохса
def boxcox_transform(data):
    # Применяем преобразование Бокса-Кокса
    transformed_data, lambda_value = boxcox(data + 0.001)
    return transformed_data, lambda_value


# Распределение признаков после преобразования, а также стат тест на нормальность распределения
def plot_transformed(data, columns):

    for column in columns:
        # Применяем преобразование Йео-Джонсона
        transformed_data_yeojohson, lambda_yeojohnson = yeo_johnson_transform(data[column])

        # Применяем преобразование Бокса-Кокса
        transformed_data_boxcox, lambda_boxcox = boxcox_transform(data[column])

        # Тесты на нормальность Шапиро-Уилка и Колмогорова-Смирнова:
        stat_shapiro, p_value_shapiro = shapiro(data[column])
        stat_kstest, p_value_kstest = kstest(data[column], "norm")

        stat_shapiro_yeojonhnson, p_value_shapiro_yeojonhnson = shapiro(transformed_data_yeojohson)
        stat_kstest_yeojonhnson, p_value_kstest_yeojonhnson = kstest(
            transformed_data_yeojohson, "norm"
        )

        stat_shapiro_boxcox, p_value_shapiro_boxcox = shapiro(transformed_data_boxcox)
        stat_kstest_boxcox, p_value_kstest_boxcox = kstest(transformed_data_boxcox, "norm")

        # Построение графиков

        plt.figure(figsize=(24, 6))

        # Распределение исходных данных
        plt.subplot(1, 3, 1)
        plt.hist(x=data[column], bins=50, color="skyblue", edgecolor="black")
        plt.title(f"Исходное распределение {column}")
        plt.xlabel(f"Значение {column}")
        plt.ylabel(f"Частота")
        plt.text(
            0.5,
            0.95,
            f"Shapiro: stat={stat_shapiro:.3f}, p-value={p_value_shapiro:.3f}\
                \nKS:stat={stat_kstest:.3f}, p-value={p_value_kstest:.3f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            fontsize=10,
        )

        # Распределение после преобразования Йео-Джонсона
        plt.subplot(1, 3, 2)
        plt.hist(x=transformed_data_yeojohson, bins=50, color="lightgreen", edgecolor="black")

        plt.title(f"Преобразование Йео-Джонсона (lambda={lambda_yeojohnson:.2f})")
        plt.xlabel(f"Преобразованное значение {column}")
        plt.ylabel(f"Частота")

        plt.text(
            0.5,
            0.95,
            f"Shapiro: stat={stat_shapiro_yeojonhnson:.3f}, p-value={p_value_shapiro_yeojonhnson:.3f}\
                \nKS: stat={stat_kstest_yeojonhnson:.3f}, p-value={p_value_kstest_yeojonhnson:.3f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            fontsize=10,
        )

        # Распределение после преобразования Бокса-Кокса
        plt.subplot(1, 3, 3)
        plt.hist(x=transformed_data_boxcox, bins=50, color="salmon", edgecolor="black")

        plt.title(f"Преобразование Бокса-Кокса(lambda={lambda_boxcox:.2f})")
        plt.xlabel(f"Преобразованное значение {column}")
        plt.ylabel(f"Частота")

        plt.text(
            0.5,
            0.95,
            f"Shapiro: stat={stat_shapiro_boxcox:.3f}, p-value={p_value_shapiro_boxcox:.3f}\
                \nKS: stat={stat_kstest_boxcox:.3f}, p-value={p_value_kstest_boxcox:.3f}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()
