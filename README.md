# Attention dan Blok Transformer Sederhana

Laporan jawaban tugas Deep Learning. Seluruh kode ada di `attention_transformer.ipynb` (dapat dijalankan di Google Colab dengan **Runtime → Run all**, CPU cukup). Semua angka di bawah berasal dari hasil eksekusi notebook tersebut dengan `seed = 42`.

---

## 1. Scaled Dot-Product Attention

Data: `q = [1,0,1,0]`, `k1 = [1,0,0,0]`, `k2 = [0,1,0,0]`, `v1 = [1,0]`, `v2 = [0,1]`, sehingga `d_k = 4` dan `√d_k = 2`.

**Perhitungan manual**

1. Skor mentah: `q·k1 = 1`, `q·k2 = 0`.
2. Skor terskala: `s1 = 1/2 = 0,5`, `s2 = 0/2 = 0`.
3. Softmax: `e^0,5 = 1,6487`, `e^0 = 1`, jumlah `2,6487`.
   - `w1 = 1,6487 / 2,6487 = 0,6225`
   - `w2 = 1 / 2,6487 = 0,3775`
4. Keluaran: `o = w1·v1 + w2·v2 = [0,6225 ; 0,3775]`.

**Hasil program:** `[0.6225, 0.3775]`. Cocok dengan hitungan manual (`assert` lulus, toleransi 1e-6).

---

## 2. Multi-Head Attention (d_model = 8, h = 2)

Tiap head memakai `d_k = d_v = 8/2 = 4`. Ada empat proyeksi linear (`W_Q`, `W_K`, `W_V`, `W_O`), masing-masing 8×8 dengan bias.

| Proyeksi | Bobot | Bias | Total |
|---|---|---|---|
| W_Q | 64 | 8 | 72 |
| W_K | 64 | 8 | 72 |
| W_V | 64 | 8 | 72 |
| W_O | 64 | 8 | 72 |
| **Total** | 256 | 32 | **288** |

Program melaporkan **288** parameter. Cocok dengan hitungan manual.

Verifikasi tambahan: bobot disalin ke `nn.MultiheadAttention` bawaan PyTorch, dan keluaran serta bobot attention per head sama (tanpa mask maupun dengan causal mask).

---

## 3. Masking

Konvensi: mask boolean, `True` = posisi boleh diperhatikan. Posisi terlarang diisi `-inf` sebelum softmax.

- **Padding mask**, shape `(B,1,1,T)`: key berupa token PAD ditutup.
- **Causal mask**, shape `(1,1,T,T)`: segitiga bawah, posisi `i` hanya melihat `j ≤ i`.

Hasil pengujian (tiga jenis mask: padding, causal, padding + causal): bobot attention pada seluruh posisi terlarang bernilai **tepat 0** setelah softmax, dan setiap baris yang punya minimal satu posisi valid berjumlah 1.

**Kasus tepi (baris tanpa posisi valid):** versi naif menghasilkan `NaN` (0/0). Perbaikan memakai `torch.nan_to_num` sehingga baris tersebut bernilai 0 tanpa NaN, sedangkan baris valid tetap berjumlah 1.

---

## 4. Sinusoidal Positional Encoding

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- Shape untuk `T = 4`, `d_model = 8`: `(4, 8)`. Setelah ditambahkan ke input: `(B, T, 8)`.
- Posisi 0 menghasilkan `[0, 1, 0, 1, 0, 1, 0, 1]`.
- Nilai untuk posisi 1 diverifikasi terhadap rumus (lulus).
- PE disimpan sebagai *buffer*, bukan parameter (0 parameter, tidak dilatih).

**Fungsi:** attention tidak mengetahui urutan token (permutation-equivariant). Positional encoding ditambahkan ke embedding sehingga setiap posisi punya representasi unik. Frekuensi tiap pasangan dimensi berbeda (cepat sampai lambat), dan `PE(pos+k)` merupakan transformasi linear (rotasi) dari `PE(pos)`, sehingga relasi jarak relatif mudah dipelajari.

---

## 5. Blok Transformer

Susunan (post-LN seperti makalah 2017):

```
x → MHA → (+x) → LayerNorm → FFN → (+masukan FFN) → LayerNorm → keluaran
```

FFN: `Linear(8,16) → ReLU → Linear(16,8)` (`d_ff = 16`). Seluruh tahap menghasilkan tensor `(B, T, 8)` dan bobot attention `(B, h, T, T)`.

---

## 6. Uji Fungsional: Klasifikasi Sekuens Dua Kelas

**Tugas sintetis (bergantung urutan):** tiap sekuens berisi token A (id 1) dan B (id 2) tepat satu kali, sisanya token pengisi acak (id 3-9). Panjang 6-8, sisanya PAD (id 0). Kelas 1 jika A muncul **sebelum** B, kelas 0 jika sebaliknya. Komposisi token kedua kelas identik, sehingga model harus memakai informasi posisi.

**Konfigurasi model**

| Item | Nilai |
|---|---|
| Vocabulary | 10 |
| d_model / heads / d_ff | 8 / 2 / 16 |
| Jumlah blok | 1 |
| Pooling | mean pooling hanya atas token valid (non-PAD) |
| Dropout | 0 |
| Total parameter | 698 |

**Pelatihan:** Adam, lr = 3e-3, batch 64, 60 epoch. Data: train 3000, validasi 500, test 500 (proporsi kelas 1 pada train ≈ 0,50).

**Hasil**

| Metrik | Nilai |
|---|---|
| Val acc epoch 1 | 0,726 |
| Val acc epoch 60 | 1,000 |
| Test loss | ≈ 0,0000 |
| **Test accuracy** | **1,000** (acak = 0,500) |

**Ablasi tanpa positional encoding:** akurasi test **0,512** (≈ acak), sehingga terbukti informasi posisi diperlukan untuk tugas ini. Akurasi sempurna dengan PE wajar karena tugas sintetis ini sederhana.

---

## 7. Verifikasi Implementasi

### 7.1 Shape pada setiap tahap (B = 2, T = 4, d_model = 8, h = 2, d_k = 4)

| Tensor | Shape |
|---|---|
| Input x | (2, 4, 8) |
| Q (sebelum split head) | (2, 4, 8) |
| Query / Key / Value per head | (2, 2, 4, 4) |
| Matriks skor `QKᵀ/√d_k` | (2, 2, 4, 4) |
| Bobot attention (softmax) | (2, 2, 4, 4) |
| Konteks per head (`w @ V`) | (2, 2, 4, 4) |
| Gabungan head (concat) | (2, 4, 8) |
| Keluaran akhir MHA (setelah `W_O`) | (2, 4, 8) |

Seluruhnya diperiksa dengan `assert` di notebook.

### 7.2 Jumlah parameter per komponen

| Komponen | Manual | Program |
|---|---|---|
| Embedding (10×8) | 80 | 80 |
| Positional encoding | 0 | 0 |
| MHA (4 × (8·8+8)) | 288 | 288 |
| LayerNorm 1 (γ + β) | 16 | 16 |
| FFN ((8·16+16) + (16·8+8)) | 280 | 280 |
| LayerNorm 2 (γ + β) | 16 | 16 |
| Head klasifikasi (8·2+2) | 18 | 18 |
| **Total model** | **698** | **698** |

Satu blok transformer: 288 + 16 + 280 + 16 = **600** parameter.

### 7.3 dan 7.4 Softmax dan masking

- Setiap baris attention yang punya ≥ 1 posisi valid berjumlah 1 (tanpa mask, padding, causal, padding + causal).
- Bobot posisi terlarang bernilai tepat 0 pada semua jenis mask.
- **Causal:** mengubah token di posisi 4-5 tidak mengubah keluaran posisi 0-3. Tanpa causal mask, keluaran posisi 0-3 ikut berubah (kebocoran masa depan terbukti).
- **Padding:** mengubah isi posisi PAD tidak mengubah keluaran posisi valid.

### 7.5 Overfit satu batch kecil

Satu batch tetap berisi 16 sampel, Adam lr = 1e-2, 1000 langkah. Loss turun dari sekitar ln 2 (≈ 0,693) ke **0,00003**, akurasi **100%**.

---

## Jawaban Pertanyaan Analisis

### 1. Shape tensor (d_model = 8, h = 2, T = 4, B = 2, d_k = 4)

| Tensor | Shape |
|---|---|
| Input | (2, 4, 8) |
| Query / Key / Value setelah proyeksi linear | (2, 4, 8) |
| Query / Key / Value per head | (2, 2, 4, 4) |
| Matriks skor | (2, 2, 4, 4) |
| Keluaran akhir | (2, 4, 8) |

### 2. Mengapa skor dibagi √d_k?

Jika komponen `q` dan `k` bebas dengan rata-rata 0 dan varians 1, maka `q·k = Σ qᵢkᵢ` memiliki varians `d_k` (simpangan baku √d_k). Pembagian dengan √d_k mengembalikan varians skor ke ≈ 1 berapa pun `d_k`.

Tanpa pembagian, softmax menjadi jenuh: satu posisi mendekati probabilitas 1 dan lainnya mendekati 0 (hampir one-hot), sehingga gradien `pᵢ(1−pᵢ)` mendekati 0 dan pelatihan macet. Hasil eksperimen di notebook (10 posisi, entropi maksimum ln 10 = 2,303):

| d_k | Skala | Std skor | Rata-rata prob. maks | Rata-rata entropi |
|---|---|---|---|---|
| 64 | tanpa pembagian | 7,94 | 0,864 | 0,349 |
| 64 | dibagi √d_k | 0,99 | 0,317 | 1,931 |
| 512 | tanpa pembagian | 22,80 | 0,953 | 0,115 |
| 512 | dibagi √d_k | 1,01 | 0,322 | 1,919 |

### 3. Padding mask vs causal mask

- **Padding mask** menutup token PAD (pengisi agar panjang sekuens dalam batch sama). Bentuknya berbeda per sampel dan dipakai pada model apa pun.
- **Causal mask** menutup semua posisi masa depan, sehingga token `i` hanya melihat `j ≤ i`. Bentuknya tetap, hanya bergantung pada `T`.
- Keduanya dapat digabung dengan operasi AND.

**Jika causal mask tidak dipakai pada model autoregresif:** saat pelatihan (teacher forcing) posisi `i` dapat melihat token target di posisi berikutnya, sehingga model hanya menyalin dan loss turun semu. Saat generasi, token masa depan belum ada dan model gagal (ketidakcocokan train dan inferensi). Terbukti pada 7.3: tanpa causal mask, keluaran posisi awal berubah ketika posisi akhir diubah.

### 4. Fungsi positional encoding dan perbandingan

**Fungsi:** menyuntikkan informasi urutan, karena attention tidak mengetahui urutan token. Ablasi pada bagian 6 menunjukkan tanpa PE akurasi hanya 0,512.

| | Sinusoidal | Learned embedding |
|---|---|---|
| Parameter | 0 (fungsi tetap) | `max_len × d_model` |
| Posisi di luar panjang latih | Dapat dihitung untuk posisi sembarang (ekstrapolasi praktis terbatas) | Tidak ada embedding untuk posisi > `max_len` |
| Informasi relatif | `PE(pos+k)` transformasi linear dari `PE(pos)` | Harus dipelajari dari data |
| Fleksibilitas | Terbatas pada pola sin/cos | Menyesuaikan data |
| Contoh penggunaan | Transformer 2017 | BERT, GPT-2 |

Pada makalah asli, kualitas keduanya hampir sama.

### 5. Mengapa LayerNorm lebih sesuai daripada BatchNorm?

- BatchNorm menghitung statistik lintas batch dan lintas posisi. Pada data sekuens, panjang bervariasi dan ada token PAD sehingga statistik tercemar dan bergantung komposisi batch.
- BatchNorm bergantung pada ukuran batch, butuh *running statistics* yang berbeda antara training dan inferensi, dan hampir tidak bermakna pada generasi autoregresif (1 token per langkah).
- LayerNorm menormalkan fitur per token, tidak bergantung pada ukuran batch, panjang, atau padding, dan berperilaku sama saat training dan inferensi.

### 6. Satu kesalahan implementasi dan verifikasi perbaikannya

**Kesalahan:** pada versi naif, baris attention yang seluruh posisinya di-mask (semua `-inf`) menghasilkan `NaN` setelah softmax (0/0), lalu merambat ke loss dan gradien. Ini ditampilkan pada versi naif sebagai demonstrasi, bukan bug yang muncul tidak sengaja.

**Perbaikan:** `torch.nan_to_num(w, nan=0.0)` setelah softmax.

**Verifikasi (bagian 3 notebook):** versi naif menghasilkan NaN, versi perbaikan tidak. Baris valid tetap berjumlah 1 dan baris tanpa posisi valid bernilai 0. Semuanya dicek dengan `assert`.

### 7. Masalah model recurrent yang ditangani attention

**Masalah:** pada RNN/LSTM, informasi token jauh harus melewati banyak langkah berurutan lewat satu hidden state, sehingga terjadi *vanishing gradient* dan bottleneck vektor berukuran tetap. Komputasi juga sekuensial dan tidak dapat diparalelkan sepanjang waktu.

**Penanganan:** setiap posisi membaca setiap posisi lain secara langsung dalam satu langkah (jalur gradien `O(1)` dibanding `O(T)`), dan seluruh posisi dihitung paralel sebagai perkalian matriks.

**Hubungan dengan implementasi:** matriks skor `(B, h, T, T)` pada `scaled_dot_product_attention` menghubungkan semua pasangan posisi secara langsung. Pada tugas bagian 6, token A dan B saling melihat walau terpisah jauh. Karena jalur langsung ini tidak memuat urutan, ia dilengkapi `SinusoidalPositionalEncoding`, sedangkan causal mask menggantikan sifat alami RNN yang hanya melihat masa lalu.

### 8. Causal mask vs attention dua arah, dan komponen yang bertahan sejak 2017

| Keluarga | Attention | Konsekuensi |
|---|---|---|
| Decoder-only (GPT, LLaMA) | Causal (satu arah) | Cocok untuk generasi autoregresif. Semua posisi dilatih paralel dengan target token berikutnya, dan inferensi dapat memakai KV cache. Representasi tidak melihat konteks kanan. |
| Encoder-only (BERT) | Dua arah | Representasi memakai konteks kiri dan kanan, kuat untuk klasifikasi, NER, retrieval. Tidak menghasilkan teks token demi token secara langsung (dilatih dengan masked language modelling). |
| Encoder-decoder (Transformer 2017, T5) | Encoder dua arah, decoder causal + cross-attention | Cocok untuk terjemahan dan peringkasan. |

**Dua komponen yang bertahan sejak 2017:**
1. Multi-head scaled dot-product self-attention.
2. Residual connection yang dipasangkan dengan normalisasi layer di sekitar tiap sub-lapisan (bersama FFN per posisi).

Bentuknya berkembang (pre-LN, RMSNorm, GQA, RoPE, dan lain-lain), tetapi konsep dasarnya tetap.

### 9. Pengayaan: Rotary Position Embedding (RoPE)

**Gagasan:** pada implementasi ini posisi **ditambahkan** ke embedding input sekali di awal. Pada RoPE, vektor `q` dan `k` di setiap layer **diputar** per pasangan dimensi dengan sudut `m·θᵢ` (m = posisi), sehingga dot product `⟨R_m q, R_n k⟩` hanya bergantung pada selisih posisi `m − n`.

**Perbaikan dibanding komponen dasar:**
- Informasi posisi relatif menjadi sifat langsung dari skor attention, bukan sesuatu yang harus diekstrak jaringan dari penjumlahan.
- Tanpa parameter tambahan, dan diterapkan di tiap layer sehingga informasi posisi tidak larut di lapisan dalam.
- Perilaku terhadap panjang konteks lebih baik, dan konteks dapat diperpanjang dengan penskalaan frekuensi (mis. position interpolation).

**Verifikasi numerik di notebook:** untuk pasangan `(m, n)` = (2,0), (10,8), (103,101) (selisih sama = 2), skor RoPE identik: **−3,413995**. Sebaliknya, pada PE sinusoidal yang ditambahkan ke input, skor untuk pasangan yang sama berbeda-beda (−0,630; −1,125; −2,100), sehingga tidak murni relatif.

---

## Ringkasan Artefak

| Artefak | Lokasi |
|---|---|
| Kode attention, MHA, positional encoding, blok transformer | `attention_transformer.ipynb`, bagian 1, 2, 4, 5 |
| Skrip verifikasi terhadap hitungan manual | Bagian 1, 2, 4 |
| Tabel shape dan jumlah parameter | Bagian 7.1 dan 7.2 (juga di README ini) |
| Bukti padding mask dan causal mask | Bagian 3 dan 7.3 |
| Bukti overfit satu batch | Bagian 7.5 |
| Hasil pelatihan dan evaluasi model kecil | Bagian 6 |
| Jawaban analisis 1-9 | Bagian 8 notebook dan README ini |
