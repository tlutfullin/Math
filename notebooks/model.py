from typing import List, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor


def encoding(
    data: pd.DataFrame,
    categorical_features: Union[List[str], None] = None,
    encoder_method: Union[List[str], None] = ["ordinal", "one_hot"],
    delete_data: bool = True,
) -> pd.DataFrame:
    """
    Эта функция кодирует категориальные особенности данного кадра данных, используя указанные методы кодирования.

    Аргументы:
        data: (pd.DataFrame): входной кадр данных
        categorical_features: (Union[List[str], None]): список категориальных функций для кодирования. Если None, все категориальные функции в кадре данных будут закодированы.
        encoder_method: (Union[List[str], None]): список используемых методов кодирования. Допустимые параметры: «ordinal» и «one_hot». Если None, будет использоваться OrdinalEncoder.
        save_path: (Union[str, None]): путь для сохранения обученных кодировщиков. Если None, кодировщики не будут сохранены.
        delete_data: (bool): флаг, указывающий на удаление категориальных данных. Если True, то будет возвращен преобразованный датафрейм без категориальных переменных .

    Возвращает:
        pd.DataFrame: закодированный кадр данных.

    Возникает:
        Исключение: если неправильно написан метод кодирования или во время обработки возникает ошибка.
    """

    # из датафрейма отбираем все категориальные данные
    if categorical_features is None:
        categorical_features = data.select_dtypes(include=["category", "object"]).columns.tolist()

        categorical_features = ["brands", "manufacturer"]

    try:

        # создаем пустой датафрейм, который будет содержать закодированные данные
        encoded_data = pd.DataFrame(index=data.index)

        for feature in categorical_features:

            # проверяем какой метода кодирования использовать
            if "ordinal" in encoder_method:
                # для признаков, которые не попали в обучение будет ставится (unknown_value=-1)
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,  # устанавливает закодированное значение для неизвестных категорий.
                    encoded_missing_value=-2,  # для пропущенных(None) меток будет стоять значение -2
                    dtype=np.int64,
                )

            elif "one_hot" in encoder_method:
                encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.int64)
            else:
                raise ValueError("Unsupported encoding method")

            # Преобразование категориального признака
            encoded_feature = encoder.fit_transform(data[[feature]])

            # преобразуем разряженную матрицу к DataFrame
            if sparse.issparse(encoded_feature):
                if "one_hot" in encoder_method:
                    encoded_feature_names = [
                        f"{feature}_{x}" for x in encoder.get_feature_names_out()
                    ]
                    encoded_feature_df = pd.DataFrame(
                        encoded_feature.toarray(), index=data.index, columns=encoded_feature_names
                    )

                else:
                    encoded_feature_df = pd.DataFrame(
                        encoded_feature.toarray(), index=data.index, columns=[feature]
                    )

            else:
                encoded_feature_df = pd.DataFrame(
                    encoded_feature, index=data.index, columns=[feature]
                )

            # Добавляем закодированный признак к общему DataFrame
            encoded_data = pd.concat([encoded_data, encoded_feature_df], axis=1)

        # Конкатенируем закодированные признаки с исходными данными
        data = pd.concat([data, encoded_data], axis=1)

        if delete_data:
            # удаляем исходные категориальные переменные
            data.drop(columns=categorical_features, inplace=True)

        return data

    except Exception as e:
        print(e)
        return data


def linear_regression(X, y):
    """
    Функция для оценки модели линейной регрессии методом МНК,
    подготовки репорта модели и интерпретации оценок коэффициентов.

    Аргументы:
    data (DataFrame): Исходные данные.
    target_col (str): Название целевой переменной.
    feature_cols (list): Список названий признаков.

    Возвращает:
    summary (Summary): Сводка модели линейной регрессии.
    """
    # Добавляем константу к признакам
    X = sm.add_constant(X)

    # Оцениваем модель линейной регрессии
    model = sm.OLS(y, X).fit()

    # Выводим сводку модели
    summary = model.summary()

    return model, summary


def test_heteroskedasticity(model):
    """
    Функция для тестирования на гетероскедастичность.

    Аргументы:
    model: Модель линейной регрессии.

    Возвращает:
    results (dict): Результаты тестов на гетероскедастичность.
    """
    # Получаем остатки модели
    residuals = model.resid

    # Тестируем на гетероскедастичность тестом Бройша-Пагана
    bp_lm, bp_p_value, bp_f_value, bp_f_p_value = het_breuschpagan(residuals, model.model.exog)

    # Тестируем на гетероскедастичность тестом Уайта.
    white_lm, white_p_value, white_f_value, white_f_p_value = het_white(residuals, model.model.exog)

    results = {
        "Breusch-Pagan Test": {
            "LM Statistic": bp_lm,
            "LM-Test p-value": bp_p_value,
            "F Statistic": bp_f_value,
            "F-Test p-value": bp_f_p_value,
        },
        "White Test": {
            "LM Statistic": white_lm,
            "LM-Test p-value": white_p_value,
            "F Statistic": white_f_value,
            "F-Test p-value": white_f_p_value,
        },
    }

    return results


def test_multicollinearity(X):
    """
    Функция для тестирования на мультиколлинеарность.

    Аргументы:
    data (DataFrame): Исходные данные.
    feature_cols (list): Список названий признаков.

    Возвращает:
    vif_results (DataFrame): Результаты теста VIF (коэффициенты VIF для каждого признака).
    """
    # Вычисляем коэффициенты VIF
    X = sm.add_constant(X)

    # теста VIF (коэффициенты VIF для каждого признака).
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # теста Толлока-Вонка (коэффициенты TW для каждого признака).
    vif["TW"] = [variance_inflation_factor(X.values, i) ** 2 for i in range(X.shape[1])]

    return vif


def weighted_least_squares(X, y):
    # Добавляем столбец с константой для учета свободного члена
    X_with_intercept = sm.add_constant(X)

    # дисперсия каждого наблюдения
    model = sm.OLS(y, X_with_intercept)
    results = model.fit()
    residuals = results.resid
    squared_residuals = residuals**2
    weights = 1.0 / squared_residuals

    # параметры модели с использованием взвешенных наименьших квадратов
    wls_model = sm.WLS(y, X_with_intercept, weights=weights)
    wls_results = wls_model.fit()

    return wls_results


def robust_standard_errors(model):
    """
    Функция для расчета робастных стандартных ошибок.

    Аргументы:
    model: Модель линейной регрессии, уже оцененная.

    Возвращает:
    robust_se (array): Робастные стандартные ошибки коэффициентов модели.
    """
    robust_se = model.get_robustcov_results(cov_type="HC3").bse
    return robust_se


def metric_model(y_pred, y_true):

    res_metric = {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2-score": r2_score(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
    }

    df_metric = pd.DataFrame(data=res_metric, index=["Metrics"])

    return df_metric.T
