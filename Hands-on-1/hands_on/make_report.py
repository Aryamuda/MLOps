"""Build the submission report from the persisted experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def build_report() -> Path:
    summary_path = RESULTS / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run `python -m hands_on.run_experiments` first.")
    summary = json.loads(summary_path.read_text())
    trials = pd.read_csv(RESULTS / "search_trials.csv")
    seeds = pd.read_csv(RESULTS / "seed_repetitions.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(trials.learning_rate, trials.accuracy, c=trials.channels, cmap="viridis", s=55)
    ax.set_xscale("log")
    ax.set_xlabel("Laju pembelajaran (skala logaritmik)")
    ax.set_ylabel("Akurasi validasi terbaik")
    ax.set_title("20 percobaan pencarian acak")
    fig.colorbar(ax.collections[0], ax=ax, label="Jumlah kanal CNN")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "search_trials.png", dpi=160)
    plt.close(fig)

    std = summary["seed_validation_std"]
    report = f"""# Laporan Hands-on 1 — Pelacakan Eksperimen MLflow

**Nama:** Arya Muda Siregar

**NIM:** 123450063

## Pertanyaan dan hipotesis

**Pertanyaan:** Bagaimana konfigurasi pelatihan CNN sederhana, augmentasi, dan
dropout memengaruhi kinerja validasi pada dataset angka tulisan tangan?

**Hipotesis:** CNN yang konfigurasinya disetel akan mengungguli baseline pada
data validasi. Pengulangan eksperimen akan menunjukkan variasi akibat seed,
sedangkan penghapusan augmentasi atau dropout dapat mengubah kinerja. Pemilihan
konfigurasi hanya menggunakan data validasi; data uji disimpan untuk evaluasi
akhir.

## Dataset dan protokol

 - Dataset: `load_digits` dari scikit-learn, berisi 1.797 citra skala abu-abu berukuran 8×8 piksel dan 10 kelas.
- Pembagian stratifikasi tetap: latih {json.loads((ROOT / 'data/manifest.json').read_text())['split_counts']['train']}, validasi {json.loads((ROOT / 'data/manifest.json').read_text())['split_counts']['validation']}, dan uji {json.loads((ROOT / 'data/manifest.json').read_text())['split_counts']['test']}.
- SHA-256 data: `{summary['dataset_sha256']}`; seed pembagian: {summary['split_seed']}.
- Model: CNN PyTorch dengan dua lapisan konvolusi. Metrik optimasi: akurasi validasi; metrik tambahan: loss cross-entropy dan F1 makro.
- Pencarian: {summary['search_trials']} konfigurasi acak dengan seed tetap, masing-masing dilatih selama delapan epoch. Konfigurasi dengan hasil validasi terbaik kemudian dijalankan menggunakan seed 13, 42, dan 73.
- Ablasi: rancangan faktorial 2×2 untuk kondisi augmentasi aktif/nonaktif dan dropout aktif/nonaktif. Setiap konfigurasi diulang dengan tiga seed yang sama.

## Hasil

**Akurasi validasi baseline** adalah **{summary.get('baseline_validation_accuracy', float('nan')):.3f}**.
**Akurasi validasi terbaik** dari pencarian adalah **{summary['best_search_validation']:.3f}**.
Untuk konfigurasi terpilih, akurasi validasi dari tiga seed adalah
**{summary['seed_validation_mean']:.3f} ± {std:.3f}** (rata-rata ± simpangan
baku sampel). Hasil akhir pada data uji adalah akurasi
**{summary['test_metrics']['accuracy']:.3f}**, F1 makro
**{summary['test_metrics']['macro_f1']:.3f}**, dan loss
**{summary['test_metrics']['loss']:.3f}**.

Konfigurasi terpilih:

```json
{json.dumps(summary['best_config'], indent=2)}
```

| Percobaan | Laju pembelajaran | Kanal | Dropout | Ukuran batch | Akurasi validasi | F1 makro validasi |
|---:|---:|---:|---:|---:|---:|---:|
"""
    best_rows = trials.sort_values("accuracy", ascending=False).head(5)
    for _, row in best_rows.iterrows():
        report += f"| {int(row.trial)} | {row.learning_rate:g} | {int(row.channels)} | {row.dropout:g} | {int(row.batch_size)} | {row.accuracy:.3f} | {row.macro_f1:.3f} |\n"

    report += "\n### Pengulangan seed\n\n| Seed | Akurasi validasi | F1 makro validasi | Loss validasi |\n|---:|---:|---:|---:|\n"
    for _, row in seeds.iterrows():
        report += f"| {int(row.seed)} | {row.accuracy:.3f} | {row.macro_f1:.3f} | {row.loss:.3f} |\n"
    report += "\n### Ringkasan ablasi\n\n| Augmentasi | Dropout | Rata-rata akurasi ± simpangan baku | Rata-rata F1 makro ± simpangan baku |\n|---|---:|---:|---:|\n"
    for _, row in pd.read_csv(RESULTS / "ablation.csv").groupby(["augmentation", "dropout"]):
        accuracy = row.accuracy
        f1 = row.macro_f1
        report += f"| {'Aktif' if row.augmentation.iloc[0] else 'Nonaktif'} | {row.dropout.iloc[0]:g} | {accuracy.mean():.3f} ± {accuracy.std(ddof=1):.3f} | {f1.mean():.3f} ± {f1.std(ddof=1):.3f} |\n"

    if summary.get("registered_model"):
        registry_section = f"""Model akhir dicatat dari run `{summary['final_run_id']}`, didaftarkan sebagai
`{summary['registered_model']}` versi `{summary['registered_version']}`, dan diberi
alias `@{summary['alias']}`. Model dimuat melalui
`models:/{summary['registered_model']}@{summary['alias']}` dan menghasilkan
prediksi **{summary['sample_prediction']}** untuk sampel validasi. Run tersebut
mencatat hash dataset, seed pembagian, revisi kode jika tersedia, serta versi
lingkungan."""
    else:
        registry_section = f"""Kandidat yang dilatih secara lokal menghasilkan prediksi sampel
**{summary['sample_prediction']}**. Pendaftaran model MLflow dan alias
`@champion` belum tersedia karena MLflow tidak dapat dipasang di lingkungan
ini. Setelah dependensi terpasang, jalankan `python -m hands_on.run_experiments`
dan buat ulang laporan berdasarkan hasil yang tercatat di MLflow."""
    report += f"""
### Grafik dan tangkapan layar

![Akurasi validasi pada setiap percobaan pencarian](results/search_trials.png)

![Tampilan perbandingan MLflow untuk baseline-default dan trial-16](images/mlflow-compare.png)

![Parameter dan metrik pada perbandingan MLflow](images/mlflow-compare-params-metrics.png)

![Tag run pada perbandingan MLflow](images/mlflow-compare-tags.png)

![Model registry MLflow dengan alias champion](images/model-registry.png)

## Registri model dan asal-usul eksperimen

{registry_section}

## Kesimpulan

Hasil pengamatan menunjukkan akurasi validasi terbaik sebesar
{summary['best_search_validation']:.3f}; rata-rata dari tiga seed sebesar
{summary['seed_validation_mean']:.3f} dengan simpangan baku {std:.3f}; dan
akurasi akhir pada data uji sebesar {summary['test_metrics']['accuracy']:.3f}.
Tabel ablasi merangkum dampak kedua komponen pada pembagian data ini. Hasil ini
berlaku sebagai perbandingan untuk keperluan praktikum. Dataset berukuran kecil
dan satu pembagian data uji belum cukup untuk menyimpulkan kemampuan generalisasi
secara luas atau kesiapan untuk produksi.

## Artefak untuk reproduksi

- `results/search_trials.csv`: seluruh 20 konfigurasi dan metrik validasi.
- `results/seed_repetitions.csv`: konfigurasi terpilih yang dijalankan dengan tiga seed.
- `results/ablation.csv` dan `results/ablation_summary.csv`: hasil ablasi berpasangan.
- `results/summary.json`: metrik akhir, alias, dan asal-usul model.
- `data/digits.npz.dvc` dan `data/manifest.json`: pointer versi data dan hash.
"""
    if summary.get("registered_model"):
        report += "- Metadata pelacakan MLflow: `mlflow.db`; artefak model dan run: `mlartifacts/`.\n"
    else:
        report += "- Metadata pelacakan MLflow dan artefak registri akan dibuat saat eksperimen MLflow dijalankan.\n"
    path = ROOT / "REPORT.md"
    path.write_text(report)
    return path


if __name__ == "__main__":
    print(build_report())
