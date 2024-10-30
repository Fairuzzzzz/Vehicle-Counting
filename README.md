# Vehicle Counting

## Penggunaan :

### Line Drawer :

1. Jalankan script `line_drawer.py` dengan command
```bash
python line_drawer.
```

2. Upload gambar yang akan menjadi tumpuan koordinat garisnya, kemudian save

### Vehicle Counter:
Ubah pada script `vehicle_counter.py`
```py
json_path = "british_line.json" # Update data json sesuai dengan data mu
video_path = "british_highway_traffic.mp4"  # Update video path sesuai dengan video mu
```

Ubah `json_path` sesuai dengan hasil koordinat garis yang telah di buat pada `line_drawer.py` sebelumnya.

Ubah `video_path` sesuai dengan gambar dari video yang menjadi tumpuan koordinat garis pada `line_drawer.py` sebelumnya.

2. Jalankan script
```bash
python vehicle_counter.py
```
