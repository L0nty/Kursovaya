import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style='whitegrid')


# ============================================================
# Первичный анализ временного ряда NVIDIA (NVDA)
# Датасет: NVDA_yfinance_clean.csv
# Задача: прогнозирование
# ============================================================


def load_data(path: str) -> pd.DataFrame:
    """Загрузка и базовая подготовка данных."""
    df = pd.read_csv(path)

    # Преобразование даты и установка индекса
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').set_index('Date')

    # Приведение порядка столбцов к более привычному виду
    preferred_order = ['Open', 'High', 'Low', 'Close', 'Volume']
    existing_cols = [c for c in preferred_order if c in df.columns]
    df = df[existing_cols]

    return df


def print_stage_title(title: str) -> None:
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)


def describe_sampling(index: pd.DatetimeIndex) -> None:
    """Анализ интервалов между соседними наблюдениями."""
    diffs = index.to_series().diff().dropna()
    diff_days = diffs.dt.days.value_counts().sort_index()

    print('\nРаспределение интервалов между соседними наблюдениями (в днях):')
    print(diff_days)

    if len(diff_days) == 1:
        print('\nИнтервалы полностью равномерные.')
    else:
        print(
            '\nИнтервалы не полностью равномерные по календарю: '
            'есть пропуски выходных/праздников, что нормально для биржевых данных.'
        )
        print('Фактически ряд наблюдается по торговым дням.')


def detect_outliers_3sigma(series: pd.Series) -> int:
    """Количество потенциальных выбросов по правилу трех сигм."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return 0
    mask = (series < mean - 3 * std) | (series > mean + 3 * std)
    return int(mask.sum())


def classify_snr(snr_db: float) -> str:
    if snr_db > 20:
        return 'Отлично'
    if snr_db > 10:
        return 'Хорошо'
    if snr_db > 0:
        return 'Удовлетворительно'
    return 'Плохо'


def interpret_residual_distribution(residuals: pd.Series) -> str:
    """Очень грубая эвристика для краткой текстовой интерпретации распределения остатков."""
    x = residuals.dropna()
    skew = x.skew()
    kurt = x.kurt()

    if abs(skew) < 0.5 and kurt < 1.5:
        return 'Близко к нормальному, примерно симметричное'
    if abs(skew) >= 0.5:
        return 'Асимметричное'
    if kurt >= 1.5:
        return 'С тяжелыми хвостами'
    return 'Смешанная форма, требует дополнительного анализа'


# -----------------------------
# Основной скрипт
# -----------------------------

DATA_PATH = 'NVDA_yfinance_clean.csv'
df = load_data(DATA_PATH)


# ============================================================
# ЭТАП 1. Загрузка и первичное знакомство с данными
# ============================================================
print_stage_title('ЭТАП 1. Загрузка и первичное знакомство с данными')

print('\nПервые 5 строк данных:')
print(df.head())

print('\nИнформация о данных:')
print(df.info())

print(f'\nРазмерность данных: {df.shape}')
print(f'Количество наблюдений: {len(df)}')
print(f'Количество каналов: {df.shape[1]}')
print(f'Период данных: {df.index.min().date()} — {df.index.max().date()}')

print('\nТипы данных по столбцам:')
print(df.dtypes)

print('\nВыводы по этапу 1:')
print('- Данные загружены корректно.')
print('- Ряд многомерный: Open, High, Low, Close, Volume.')
print('- Временная метка присутствует и преобразована в datetime.')
print('- Дата установлена в качестве индекса.')


# ============================================================
# ЭТАП 2. Визуализация исходных данных
# ============================================================
print_stage_title('ЭТАП 2. Визуализация исходных данных')

channels = df.columns.tolist()
fig, axes = plt.subplots(len(channels), 1, figsize=(16, 12), sharex=True)

if len(channels) == 1:
    axes = [axes]

for ax, col in zip(axes, channels):
    ax.plot(df.index, df[col], linewidth=1)
    ax.set_title(col)
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Дата')
plt.suptitle('Исходные временные ряды NVDA', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('stage2_multichannel_plot.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(16, 5))
plt.plot(df.index, df['Close'], linewidth=1.3, label='Close')
plt.title('Цена закрытия NVIDIA (Close)')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stage2_close_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nВыводы по этапу 2:')
print('- У цены закрытия наблюдается выраженный долгосрочный восходящий тренд.')
print('- Ряд не выглядит стационарным: уровень и волатильность меняются со временем.')
print('- Явной календарной сезонности по графику не видно, но есть локальные повторяющиеся колебания.')
print('- Разрывов внутри самих наблюдений визуально не видно.')


# ============================================================
# ЭТАП 3. Статистический анализ
# ============================================================
print_stage_title('ЭТАП 3. Статистический анализ')

stats_table = df.describe(percentiles=[0.25, 0.5, 0.75]).T
stats_table = stats_table.rename(columns={'25%': 'Q1', '50%': 'Q2', '75%': 'Q3'})
stats_table = stats_table[['count', 'mean', 'std', 'min', 'Q1', 'Q2', 'Q3', 'max']]

print('\nОписательные статистики:')
print(stats_table.round(4))

describe_sampling(df.index)

print('\nВыводы по этапу 3:')
print('- Стандартное отклонение у Volume намного выше по масштабу, чем у ценовых каналов.')
print('- Каналов с почти нулевым стандартным отклонением нет.')
print('- Для моделей машинного обучения масштабирование признаков желательно.')
print('- Интервалы соответствуют торговым дням, а не непрерывному ежедневному календарю.')


# ============================================================
# ЭТАП 4. Анализ пропусков и выбросов
# ============================================================
print_stage_title('ЭТАП 4. Анализ пропусков и выбросов')

missing_count = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(4)
missing_table = pd.DataFrame({
    'missing_count': missing_count,
    'missing_pct': missing_pct,
})

print('\nПропуски по каналам:')
print(missing_table)

outlier_counts = {col: detect_outliers_3sigma(df[col]) for col in df.columns}
outlier_table = pd.DataFrame({
    'outliers_3sigma_count': pd.Series(outlier_counts),
    'outliers_3sigma_pct': pd.Series(outlier_counts) / len(df) * 100,
}).round(4)

print('\nПотенциальные выбросы по правилу трех сигм:')
print(outlier_table)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    axes[i].boxplot(df[col].dropna(), vert=True)
    axes[i].set_title(col)
    axes[i].grid(True, alpha=0.3)

for j in range(len(df.columns), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Диаграммы размаха по каждому каналу', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('stage4_boxplots_by_channel.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nВыводы по этапу 4:')
print('- Пропущенных значений нет.')
print('- Потенциальные выбросы присутствуют во всех каналах, особенно в Volume.')
print('- Для биржевых данных это не обязательно ошибки: это могут быть реальные рыночные скачки.')
print('- Удалять выбросы без предметной проверки не рекомендуется.')


# ============================================================
# ЭТАП 5. Анализ диапазонов значений
# ============================================================
print_stage_title('ЭТАП 5. Анализ диапазонов значений')

plt.figure(figsize=(12, 6))
df.boxplot()
plt.title('Общая диаграмма размаха для всех каналов')
plt.ylabel('Значение')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stage5_boxplot_all_channels.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nВыводы по этапу 5:')
print('- Диапазон Volume существенно отличается от диапазонов ценовых признаков.')
print('- Для совместного использования признаков в моделях нужна стандартизация или нормализация.')
print('- Для большинства моделей временных рядов здесь разумнее начинать со стандартизации (StandardScaler).')


# ============================================================
# ЭТАП 6. Корреляционный анализ
# ============================================================
print_stage_title('ЭТАП 6. Корреляционный анализ')

corr = df.corr(method='pearson')
print('\nМатрица корреляции Пирсона:')
print(corr.round(4))

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, square=True)
plt.title('Тепловая карта корреляций')
plt.tight_layout()
plt.savefig('stage6_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

close_corr = corr['Close'].sort_values(ascending=False)
print('\nКорреляция признаков с целевой переменной Close:')
print(close_corr.round(4))

print('\nВыводы по этапу 6:')
print('- Open, High, Low и Close почти идеально коррелируют друг с другом.')
print('- Это указывает на очень сильную мультиколлинеарность ценовых признаков.')
print('- Volume коррелирует с ценой заметно слабее и даже отрицательно.')
print('- При построении модели может потребоваться отбор признаков или переход к доходностям/лагам.')


# ============================================================
# ЭТАП 7. Поиск и анализ шумов
# ============================================================
print_stage_title('ЭТАП 7. Поиск и анализ шумов')

# Ключевой канал
series = df['Close']

# Для дневных биржевых данных разумно рассмотреть годовую сезонность в торговых днях
SEASONAL_PERIOD = 252
MODEL_TYPE = 'additive'

result = seasonal_decompose(
    series,
    model=MODEL_TYPE,
    period=SEASONAL_PERIOD,
    extrapolate_trend='freq'
)

trend = result.trend
seasonal = result.seasonal
resid = result.resid

if MODEL_TYPE == 'additive':
    signal = trend + seasonal
else:
    signal = trend * seasonal

mask = signal.notna() & resid.notna()
signal_clean = signal[mask]
resid_clean = resid[mask]

signal_var = signal_clean.var()
noise_var = resid_clean.var()
snr_db = 10 * np.log10(signal_var / noise_var)
snr_quality = classify_snr(snr_db)
residual_form = interpret_residual_distribution(resid_clean)

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
axes[0].plot(series.index, series)
axes[0].set_title('Исходный ряд: Close')
axes[0].grid(True, alpha=0.3)

axes[1].plot(trend.index, trend)
axes[1].set_title('Тренд')
axes[1].grid(True, alpha=0.3)

axes[2].plot(seasonal.index, seasonal)
axes[2].set_title('Сезонная компонента')
axes[2].grid(True, alpha=0.3)

axes[3].plot(resid.index, resid)
axes[3].set_title('Остатки (шум)')
axes[3].grid(True, alpha=0.3)
axes[3].set_xlabel('Дата')

plt.suptitle('Декомпозиция ряда Close', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('stage7_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 5))
plt.hist(resid_clean.dropna(), bins=50, edgecolor='black', alpha=0.8)
plt.title('Гистограмма распределения остатков')
plt.xlabel('Остатки')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('stage7_residual_hist.png', dpi=150, bbox_inches='tight')
plt.show()

print(f'\nПериод сезонности: {SEASONAL_PERIOD} торговых дней')
print(f'Модель декомпозиции: {MODEL_TYPE}')
print(f'Дисперсия сигнала: {signal_var:.6f}')
print(f'Дисперсия шума: {noise_var:.6f}')
print(f'SNR: {snr_db:.2f} дБ')
print(f'Качественная оценка SNR: {snr_quality}')
print(f'Форма распределения остатков: {residual_form}')

print('\nВыводы по этапу 7:')
print('- Тренд выраженно восходящий.')
print('- На выбранном периоде видна долгосрочная сезонная компонента, но для биржевых рядов ее интерпретация ограничена.')
print(f'- Отношение сигнал/шум: {snr_db:.2f} дБ ({snr_quality}).')
print(f'- Остатки: {residual_form}.')
print('- При необходимости сглаживания можно проверить скользящее среднее или медианный фильтр, но без агрессивной очистки.')


# ============================================================
# ИТОГ
# ============================================================
print_stage_title('ИТОГОВЫЕ ВЫВОДЫ')
print('- Датасет подходит для первичного анализа и последующей задачи прогнозирования.')
print('- Пропусков нет, качество данных хорошее.')
print('- Ценовые признаки почти дублируют друг друга по информации.')
print('- Перед моделированием желательно сформировать лаговые признаки и/или доходности.')
print('- Для моделей, чувствительных к масштабу, следует применять масштабирование.')
