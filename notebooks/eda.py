import numpy as np
import pandas as pd


def calculate_psi(expected: pd.DataFrame, actual: pd.DataFrame, buckets: int = 10) -> float:
    # Функция для расчета PSI между двумя распределениями

    # Делим переменные на корзины (buckets)
    def create_bucket_array(arr, buckets):
        breakpoints = np.arange(0, buckets + 1) / buckets * 100
        return np.percentile(arr, breakpoints)

    # Расчет PSI для одной переменной
    def calculate_variable_psi(expected_array, actual_array, buckets):
        expected_percentiles = create_bucket_array(expected_array, buckets)
        actual_percentiles = create_bucket_array(actual_array, buckets)

        expected_counts = np.histogram(expected_array, expected_percentiles)[0]
        actual_counts = np.histogram(actual_array, actual_percentiles)[0]

        expected_counts_pct = expected_counts / len(expected_array)
        actual_counts_pct = actual_counts / len(actual_array)

        # Добавляем небольшое значение для избежания деления на ноль
        actual_counts_pct[actual_counts_pct == 0] = 0.001

        psi = np.sum(
            (expected_counts_pct - actual_counts_pct)
            * np.log(expected_counts_pct / actual_counts_pct)
        )
        return psi

    # Проверка наличия NaN значений и их замена медианой
    expected = expected.fillna(expected.median())
    actual = actual.fillna(actual.median())

    # Расчет PSI для каждой переменной
    psi_values = []

    # Если переданы серии, а не фреймы данных
    if isinstance(expected, pd.Series) and isinstance(actual, pd.Series):
        psi = calculate_variable_psi(expected, actual, buckets)
        psi_values.append(psi)

    # Если переданы фреймы данных
    elif isinstance(expected, pd.DataFrame) and isinstance(actual, pd.DataFrame):
        for col in expected.columns:
            psi = calculate_variable_psi(expected[col], actual[col], buckets)
            psi_values.append(psi)

    else:
        raise ValueError("Expected and actual must both be pandas Series or DataFrames.")

    # Возвращаем PSI для всех переменных
    return psi_values
