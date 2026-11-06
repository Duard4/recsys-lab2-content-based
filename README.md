# ЛР2 — Контентно-орієнтована фільтрація (елективні курси)

Вхід: датасет і кластери курсів з ЛР1 (`data/processed`, `data/lab1_results`). Основне розбиття курсів — `cluster_C1` (k = 10).

Порядок запуску (Python 3.11; pandas, numpy, scipy, scikit-learn, matplotlib):

1. `python 07_build_pools_recommend.py` — пули №1–№4 і ТОП-10 для всіх користувачів → `results/pools/*.csv`, `results/recommendations_top10.csv`
2. `python 08_holdout_sensitivity.py` — додатково: відкладена вибірка, чутливість до кластеризації (C1/C3/C4); потрібен для `11_export_report_data.py`, у звіт не винесено
3. `python 09_distribution_stats.py` — середнє/мін/макс за кластерами для активних, пасивних, нових; графіки
4. `python 10_plot_holdout.py` — додатково: діаграма відкладеної вибірки (у звіт не винесено)
5. `python 11_export_report_data.py` — збирання даних для звіту (`results/report_data.json`)
6. `python 12_similarity_check.py` — перевірка якості контентної схожості
7. `cd report && NODE_PATH=$(npm root -g) node build_report.js` — генерація звіту `Lab2_Content_Based_Report.docx` (потрібен npm-пакет `docx`)

Ключові файли результатів: `results/pools/pool_P1.csv`, `pool_P2.csv`, `pool_P3.csv`, `pool_P4_{20,50,100,200,500}.csv`;
`results/recommendations_top10.csv`; `results/stats_*.csv`; `results/new_users_seed_courses.csv` (насіння холодного старту для нових користувачів).
