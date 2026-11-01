# Archery Form Lab (panahan tradisional) — YOLO + MoveNet, GUI web

    pip install -r requirements.txt
    python web.py            # lalu buka http://localhost:8080  (HP/PC lain: http://IP-PC:8080)

Semua diatur dari browser:
- **Kamera**: tiap kamera (Depan, Samping, Target) bisa USB / WiFi (HTTP-MJPEG) / RTSP / File video / **Tidak dipakai**.
  Target "Tidak dipakai" = hanya analisis pose. Ada tombol Cari USB dan Uji koneksi.
- **Target**: ambil gambar, klik pusat lalu tepi luar, simpan. Penilaian cincin atau kena/meleset (bandul).
- **Acuan**: target ± toleransi tiap metrik (kosong = rata-rata pribadi). "Jadikan acuan" pada tembakan terbaik.
- **Live / Hasil / Riwayat**: video beranotasi, fase tembakan, skor bentuk, skor panah, sebaran, laporan HTML/CSV.

## Cara kerja
- YOLO mendeteksi pemanah, MoveNet membaca 17 titik tubuh (cadangan: pose.backend = yolo_pose).
- Fase tembakan dari jarak dua pergelangan (satuan torso) dan kecepatan tangan tarikan: tarik -> tahan -> lepas -> lanjutan.
- Metrik bentuk (median saat tahan penuh) dibandingkan acuan; catatan dibuat otomatis.
- Kamera target: selisih gambar saat adegan diam -> anak panah baru -> skor dari jarak ke pusat.
- Pose di-sync dengan target lewat waktu lepas tali.

## Catatan
- Nilai acuan awal adalah CONTOH. Kalibrasi dengan pelatih / tembakan acuan.
- Pose 2D dari satu kamera cocok untuk membandingkan antar tembakan, bukan sudut sebenarnya.
- Kamera harus tetap. Kamera target tegak lurus papan lebih akurat.
- Hasil tiap sesi: data/<waktu>/ (shots/, shots.csv, summary.txt/json, report.html, target_final.jpg).

## Analisa rekaman video (tab "Rekaman")
Unggah video dari browser (atau ketik lokasi file): kamera depan/samping untuk pose, rekaman target untuk skor.
- Jenis: Pose + skor | Pose saja | Skor saja.
- Pencocokan skor: Urutan (rekaman tak perlu sinkron) atau Waktu (isi offset detik tiap rekaman).
- Berjalan secepat mungkin dengan bilah progres; hasil di tab Hasil/Riwayat (laporan HTML + CSV).
- Kamera langsung dari NVR/DVR: tab Kamera -> RTSP.
