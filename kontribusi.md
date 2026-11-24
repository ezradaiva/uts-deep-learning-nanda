# Kontribusi Anggota dan Penggunaan AI

Dokumen ini merangkum kontribusi yang tercatat di repository dan bantuan AI yang digunakan. Setiap anggota tetap bertanggung jawab memahami kode, hasil, dan laporan yang dikumpulkan.

## Identitas, kontribusi, dan pull request

| Anggota | NIM | Akun GitHub | Kontribusi yang tercatat | Pull request ke repository kelompok |
|---|---:|---|---|---|
| Nanda Pratama Itma | 101032330212 | `nandapratama823` | Menyiapkan repository kelompok dan alur Google Colab; mengunggah notebook, kode, CSV hasil, dan laporan; serta mengirim undangan organisasi kepada Ezra. | [PR #2](https://github.com/nanda-ezra-deeplearning/uts-deep-learning-nanda/pull/2) — **merged**. PR dibuat dari branch `perbaikan-laporan` pada repository organisasi, bukan dari fork pribadi Nanda. |
| Ezra Daiva Keitaro | 1103223200 | `ezradaiva` | Meninjau notebook dan laporan, memeriksa format file hasil, serta menyampaikan usulan perbaikan. | [PR #1](https://github.com/nanda-ezra-deeplearning/uts-deep-learning-nanda/pull/1) — **merged** dari fork Ezra. |

## Penggunaan AI

- **Alat:** OpenAI Codex.
- **Bagian yang dibantu:** membaca instruksi UTS dan datasheet; membantu menyusun pipeline PyTorch untuk klasifikasi multimodal, prediksi Remaining Useful Life (RUL), dan clustering; menyiapkan notebook Colab; serta membantu menyusun README, analisis hasil, dan draf laporan.
- **Pemeriksaan yang tercatat:** CSV prediksi berisi 16.000 baris klasifikasi, 300 baris RUL, dan 36.000 baris clustering. Pemeriksaan jumlah baris dan format tidak membuktikan skor hidden test.
- **Batas evaluasi:** label hidden test klasifikasi dan label acuan material tidak tersedia dalam arsip yang diperiksa. Karena itu, macro-F1 hidden test dan ARI tidak bisa dihitung secara lokal.
- **Tanggung jawab kelompok:** anggota harus meninjau hasil, memahami metode, dan mampu menjelaskan proyek saat ujian lisan.

## Persetujuan penggunaan AI

Status berikut tercatat pada dokumen kontribusi sebelumnya. Setiap anggota perlu memastikan sendiri bahwa persetujuan dan tanggalnya benar.

| Anggota | Persetujuan penggunaan AI | Tanggal |
|---|---|---|
| Nanda Pratama Itma (101032330212) | Setuju | 20-11-2026 |
| Ezra Daiva Keitaro (1103223200) | Setuju | 20-11-2026 |

## Status kelengkapan

- Berkas utama yang dipersyaratkan sudah tersedia di branch `main`: `laporan.pdf`, folder `kode/`, tiga CSV prediksi, dan `kontribusi.md`.
- Laporan Word dan PDF diperbarui di `main` pada commit [440e9a1](https://github.com/nanda-ezra-deeplearning/uts-deep-learning-nanda/commit/440e9a10372c5f376d8cf0cc18adf23b7d9f2d54).
- PR Ezra #1 dan PR Nanda #2 sudah berstatus merged.
- **Catatan syarat fork:** PR Ezra berasal dari fork pribadi. PR Nanda #2 berasal dari branch repository organisasi, bukan fork pribadi. Jika dosen menerapkan syarat fork untuk masing-masing anggota secara ketat, kontribusi Nanda belum memenuhi bagian alur fork tersebut.
- Jalankan ulang notebook yang ada di repository dengan dataset kelompok dan pastikan angka dalam laporan cocok dengan hasilnya. Nilai macro-F1 hidden test dan ARI hanya dapat dinilai oleh pengajar yang memiliki label acuan.
