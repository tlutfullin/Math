import re

import pandas as pd


def clean_data(data: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Эта функция очищает данные, удаляя ненужные столбцы.
    очистка столбцов ["price", "bonus", "credit"] и преобразование их в числовые значения.

    Параметры:
        data (pd.DataFrame): данные, подлежащие очистке.

    Возврат:
        pd.DataFrame: очищенные данные.

    Возникает:
        Исключение: если возникла проблема с очисткой данных.
    """

    try:

        num_rows = data.shape[0]
        num_columns = data.shape[1]
        # Drop the id column
        data = data.drop(columns=["id"])
        data = data.drop_duplicates()

        # Удаляем строки и столбцы, в которых более threshold значений являются NaN
        data = data.dropna(axis=1, thresh=int(threshold * num_rows))
        data = data.dropna(axis=0, thresh=int(threshold * num_columns))

        # Делаем очистку в столбцах, превращаем их числовой тип
        for column in ["price", "bonus", "credit"]:
            data[column] = data[column].apply(
                lambda x: re.sub(r"[^\d.]+", "", str(x)) if isinstance(x, str) else x
            )

            data[column] = pd.to_numeric(data[column], errors="coerce")

        return data

    except Exception as e:
        print(e)
        return data
