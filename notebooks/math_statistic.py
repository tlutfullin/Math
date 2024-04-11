from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multicomp as mc
from scipy import stats
from scipy.stats import (
    alexandergovern,
    bootstrap,
    f_oneway,
    kendalltau,
    kruskal,
    ks_2samp,
    mannwhitneyu,
    pearsonr,
    permutation_test,
    spearmanr,
    ttest_ind,
)
from statsmodels.stats.multitest import multipletests


# точечная оценка
def stats_test_data(data: pd.DataFrame, group_column: str, metric_column: str) -> Dict:
    """
    Эта функция выполняет статистические тесты, чтобы определить, существуют ли значительные различия
    в средних значениях данного показателя в разных группах/популяциях.

    Параметры:
    data (pd.DataFrame): кадр данных, содержащий данные.
    group_column (str): имя столбца, содержащего метки групп.
    metric_column (str): имя столбца, содержащего значения метрики.

    Возврат:
    Tuple[float, float]: кортеж, содержащий результаты трех выполненных статистических тестов:
    F-тест, тест Крускала-Уоллиса и тест Александера-Говерна.
    """
    groups = data[group_column].unique()
    group_data = {
        group: data[data[group_column] == group][metric_column]
        for group in groups
        if len(data[data[group_column] == group]) > 1
    }

    # Тест ANOVA об одинаковым среднее значение по совокупности.
    f_stats, p_value_f = f_oneway(*group_data.values())

    # Тест Крускала-Уоллиса о равенстве медиан всех групп популяции
    h_stats, p_value_h = kruskal(*group_data.values())

    # Тест Александера-Говерна о равенстве k независимых средних
    alex_result = alexandergovern(*group_data.values())
    alex_stats, p_value_alex = alex_result.statistic, alex_result.pvalue

    return {
        "f_oneway": (f_stats, p_value_f),
        "kruskal": (h_stats, p_value_h),
        "alexandergovern": (alex_stats, p_value_alex),
    }


# точечная оценка
def stats_test_groups(group1, group2) -> Dict:

    # Тест Манна-Уитни
    mw_statistic, mw_p_value = mannwhitneyu(group1, group2)

    # Перестановочный тест
    res_permutation = permutation_test((group1, group2), stats_mean, n_resamples=1000)
    perm_statistic, perm_p_value = res_permutation.statistic, res_permutation.pvalue

    # T-тест для независимых выборок
    t_statistic, t_p_value = ttest_ind(group1, group2)

    # ANOVA для двух выборок
    anova_statistic, anova_p_value = f_oneway(group1, group2)

    # Множественное тестирование с использованием метода Бонферрони
    p_values = [mw_p_value, perm_p_value, t_p_value, anova_p_value]
    corrected_p_values = multipletests(p_values, method="bonferroni")[1]

    return {
        "mannwhitneyu": (mw_statistic, corrected_p_values[0]),
        "permutation": (perm_statistic, corrected_p_values[1]),
        "t_test": (t_statistic, corrected_p_values[2]),
        "f_oneway": (anova_statistic, corrected_p_values[3]),
    }


# Определение функции, вычисляющей среднее значение
def stats_mean(data1, data2):  # Принимает два аргумента - данные для брендов ASUS и Apple
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    return mean1 - mean2  # Возвращает разницу между средними значениям


# Бутстрапирование и проведение тестов
def calculate_bootstrap_and_tests(
    data: pd.DataFrame, brands_combinations: List[Tuple], column_metric: str
):

    # различие между группами
    res_groups_test = []
    res_bootstrap_list = []

    for brands1, brands2 in brands_combinations:

        group1 = data[data["brands"] == brands1][column_metric]
        group2 = data[data["brands"] == brands2][column_metric]

        result_stats_test_groups = {(brands1, brands2): stats_test_groups(group1, group2)}

        res = bootstrap(
            data=(
                data[data["brands"] == brands1][column_metric],
                data[data["brands"] == brands2][column_metric],
            ),
            statistic=stats_mean,
            n_resamples=1000,
            random_state=42,
        )

        res = {
            (brands1, brands2): {
                "ConfidenceInterval": (res.confidence_interval, res.standard_error)
            }
        }

        res_bootstrap_list.append(res)
        res_groups_test.append(result_stats_test_groups)

    return res_bootstrap_list, res_groups_test


# =========================================== Форматированный вывод тестов и бутстрапа =====================
def print_bootstrap_comparison(res_bootstrap_list: List[Dict]):

    print("   Bootstrap results for mean difference with 95% confidence    ")
    print("================================================================")
    print(" group1    group2    std_error   lower        upper       reject")
    print("----------------------------------------------------------------")

    # список отклоненных гипотез
    rejected_hypotheses = []

    for group_test in res_bootstrap_list:

        for brand, res_bootstrap in group_test.items():

            group1 = brand[0]
            group2 = brand[1]

            lower = res_bootstrap["ConfidenceInterval"][0].low
            upper = res_bootstrap["ConfidenceInterval"][0].high
            standard_error = res_bootstrap["ConfidenceInterval"][1]

            # отклонение нулевой гипотезы (если False - нулевая гипотеза верна)
            if lower <= 0 <= upper:
                reject = "False"

                # # добавляем в список неподтвержденные гипотезы
                # rejected_hypoth = {brand: res_bootstrap}
                # rejected_hypotheses.append(rejected_hypoth)

            else:
                reject = "True"

            # Форматирование строк, чтобы выровнять по ширинам
            result = "{:<10} {:<10} {:<10.3f} {:<12.3f} {:<12.3f} {:<10}".format(
                group1, group2, standard_error, lower, upper, reject
            )

            print(result)

    print("----------------------------------------------------------------")

    # return rejected_hypotheses


def print_tests_summary(res_groups_test):

    print("The results of testing hypotheses with various tests, FWER=0.05")
    print("================================================================")
    print(" group1    group2     MWU      Perm   T-test    F-oneway:")
    print("----------------------------------------------------------------")

    # список отклоненных гипотез
    rejected_hypotheses = []

    for group_test in res_groups_test:
        for brands, tests in group_test.items():

            group1 = brands[0]
            group2 = brands[1]

            mannwhitneyu = "True" if tests["mannwhitneyu"][1] > 0.05 else "False"
            permutation = "True" if tests["permutation"][1] > 0.05 else "False"
            t_test = "True" if tests["t_test"][1] > 0.05 else "False"
            f_oneway = "True" if tests["f_oneway"][1] > 0.05 else "False"
        print(
            f"{group1:<10} {group2:<10} {mannwhitneyu:<8} {permutation:<8} {t_test:<8} {f_oneway:<8}"
        )

    print("----------------------------------------------------------------")

    #     # добавляем неподтвержденные гипотезы в список
    #     if (
    #         tests["mannwhitneyu"][1] > 0.05
    #         or tests["permutation"][1] > 0.05
    #         or tests["t_test"][1] > 0.05
    #         or tests["f_oneway"][1] > 0.05
    #     ):
    #         rejected_hypoth = {brands: tests}
    #         rejected_hypotheses.append(rejected_hypoth)

    # return rejected_hypotheses


def print_scheffe_test(result_scheffe: List):

    # список отклоненных гипотез
    rejected_hypotheses = []

    print("The results of multiple testing using the Schaeffer method, FWER=0.05")
    print("================================================================")
    print(" group1    group2     stat      pval      pval_corr    reject")
    print("----------------------------------------------------------------")

    for res in result_scheffe[2]:
        group1 = res[0]
        group2 = res[1]
        reject = "True" if res[5] else "False"

        print(f"{group1:<10} {group2:<10} {res[2]:<10} {res[3]:<10} {res[4]:<10} {reject:<10}")

    #     if not reject:
    #         rejected_hypoth = {(group1, group2): 1 }
    #         rejected_hypotheses.append(rejected_hypoth)

    # return rejected_hypotheses


# ======================================= Множественное тестирования гипотез =======================


def tukeyhsd_test(data, metric_column, group_column, alpha=0.05):
    # Создаем объект MultipleComparison для выполнения Теста Тьюки

    model = mc.MultiComparison(data[metric_column], data[group_column])
    result = model.tukeyhsd(alpha=alpha)
    return result


def scheffe_test(data, metric_column, group_column, alpha=0.05):

    model = mc.MultiComparison(data[metric_column], data[group_column])
    result = model.allpairtest(stats.ttest_ind, alpha=alpha, method="bonf", pvalidx=1)

    return result


# ====================================== Корреляционные тесты =================================


def correlation_tests_with_correction(
    data: pd.DataFrame, metric1: str, metric2: str, group_column: str, alpha=0.05, method="holm"
):

    groups = data[group_column].unique()

    # Создание списков для хранения p-значений
    p_values_pearson = []
    p_values_spearman = []
    p_values_kendall = []

    results = []

    for group in groups:

        data_metric1 = data[data[group_column] == group][metric1]
        data_metric2 = data[data[group_column] == group][metric2]

        # тест Пирсона
        corr_pear, p_value_pear = pearsonr(data_metric1, data_metric2)
        p_values_pearson.append(p_value_pear)

        # Тест Спирмена
        corr_spear, p_value_spear = spearmanr(data_metric1, data_metric2)
        p_values_spearman.append(p_value_spear)

        # Тест Кендалла
        corr_kendall, p_value_kendall = kendalltau(data_metric1, data_metric2)
        p_values_kendall.append(p_value_kendall)

        res = {
            group: {
                "pearsonr": [corr_pear, p_value_pear],
                "spearmanr": [corr_spear, p_value_spear],
                "kendalltau": [corr_kendall, p_value_kendall],
            }
        }

        results.append(res)

    # Поправка на множественные сравнения методом Холма
    corrected_p_values_pearson = multipletests(p_values_pearson, method=method)[1]
    corrected_p_values_spearman = multipletests(p_values_spearman, method=method)[1]
    corrected_p_values_kendall = multipletests(p_values_kendall, method=method)[1]

    # Создание словаря с результатами

    for i, res in enumerate(results):

        for brand, tests in res.items():
            tests["pearsonr"].append(corrected_p_values_pearson[i])
            tests["spearmanr"].append(corrected_p_values_spearman[i])
            tests["kendalltau"].append(corrected_p_values_kendall[i])

    return results


def print_correlation_results(results):
    """
    Функция для форматированного вывода результатов корреляционного анализа.

    Параметры:
        results (list): Список с результатами корреляционного анализа.
    """
    header = "{:<10} {:<10} {:<10} {:<12} {:<10}".format(
        "name", "corr_stat", "pval", "pval_corr", "reject"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    for result in results:
        for group, values in result.items():
            group_name = group
            pearsonr = values["pearsonr"]
            spearmanr = values["spearmanr"]
            kendalltau = values["kendalltau"]

            reject_pearson = "True" if pearsonr[2] < 0.05 else "False"
            reject_spearman = "True" if spearmanr[2] < 0.05 else "False"
            reject_kendall = "True" if kendalltau[2] < 0.05 else "False"

            print(f"{group_name:<10}")
            reject_pearson = "True" if pearsonr[2] < 0.05 else "False"

            print(
                f"{'pearsonr':<10} {pearsonr[0]:<10.3} {pearsonr[1]:<10.3} {pearsonr[2]:<12.3} {reject_pearson:<10} "
            )

            print(
                f"{'spearmanr':<10} {spearmanr[0]:<10.3} {spearmanr[1]:<10.3} {spearmanr[2]:<12.3} {reject_spearman:<10} "
            )

            print(
                f"{'kendalltau':<10} {kendalltau[0]:<10.3} {kendalltau[1]:<10.3} {kendalltau[2]:<12.3} {reject_kendall:<10} "
            )

            print(separator)
