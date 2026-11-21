# Kode pipeline UTS Deep Learning

Notebook utama untuk pengguna adalah notebooks/uts_deep_learning.ipynb di Google Colab. Unduh notebook dan ZIP source dari repository, unggah notebook ke Colab, lalu jalankan semua sel. Colab meminta izin Drive, mencari data_uts_dl.zip pada folder yang dibagikan kelompok, dan menyimpan hasil ke MyDrive/uts_dl_colab/hasil_uts_dl.

Folder ini berisi kode pipeline untuk tiga tugas, audit dataset, dan reproduksi prediksi klasifikasi. Requirements disediakan untuk pemasangan paket tambahan di Colab. PyTorch dan Jupyter telah tersedia pada runtime Colab.

Untuk mengaudit atau mereproduksi secara lokal, arsip dataset harus tersedia pada komputer dan lingkungan Python yang sesuai. Perintah lokal tetap tersedia, tetapi bukan alur utama:

python -m kode.pipeline --data "D:\path\to\data_uts_dl.zip" --out "D:\path\to\hasil"
python -m kode.reproduce --data "D:\path\to\data_uts_dl.zip" --root "D:\path\to\hasil"

Arsip data tidak disertakan dalam ZIP source atau GitHub. Label test klasifikasi dan label referensi material tidak tersedia, sehingga hidden-test macro-F1 dan ARI tidak dapat dihitung.
