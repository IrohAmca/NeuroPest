# NeuroPest

Masaüstünde gezen, fare imlecine tepki veren bir sinek. Davranışı, gerçek sinek beyni
bağlantısından (FlyWire) çıkarılan bir devrenin simülasyonundan gelir (leaky integrate-and-fire).

**Durum (v0.3):** imleç yaklaşınca LPLC2 + LC4 nöronları uyarılır, gerçek bağlantı üzerinden
**Giant Fiber (DNp01)** ateşler, sinek uçarak kaçar; yürüme komut nöronlarına (DNp09/P9) tonik sürücü
verilince yürür. Devrenin boyutu arayüzden seçilir; her boyutun tam beyne göre doğruluğu ve hızı
ölçülmüştür.

## Çalıştırma

```bash
uv sync
uv run neuropest
```

İlk açılışta gerçek devre yoksa oyuncak devre (146 nöron) çalışır. Gerçek devre için aşağıdaki
"Gerçek veri" bölümüne bak.

- İmleç hızla yaklaşırsa sinek **uçarak kaçar**; "Ürkeklik" kaydırıcısı bu hassasiyeti ayarlar.
- "Hareketlilik" yürüme sürücüsünü ayarlar: düşükse durur, ortada arada yürür, yüksekse yürür.
- Sinek, seçilen monitörün görev çubuğunun üstündeki alanda kalır. Overlay tıklamayı geçirir.
- Kontrol penceresi: devre, boyut (seçilen katmanın ölçülmüş doğruluğu ve hızıyla), zaman adımı,
  gerçek zaman çarpanı, aktif nöron sayısı, CPU.

## Gerçek veri (FlyWire v783)

Ham dosyalar git'e girmez (`data/raw/`, `data/circuits/` yok sayılır). Kaynaklar ve git blob özetleri
(`git hash-object` ile doğrulanır):

| Dosya | Kaynak | Boyut | SHA-1 |
|---|---|---|---|
| `Connectivity_783.parquet` | github.com/philshiu/Drosophila_brain_model | 100,8 MB | `d386555d1a5f40ebfa1380bcb05b1fab044855fd` |
| `Completeness_783.csv` | aynı depo | 3,3 MB | `b5a26b82b69a3c2e7fd99cd6810c36fc8f0e492b` |
| `Supplemental_file1_neuron_annotations.tsv` | github.com/flyconnectome/flywire_annotations (`supplemental_files/`) | 31,7 MB | `02e72f6c8161d3465f77fec0edf96c5d98027a9e` |

```bash
mkdir -p data/raw && git clone --no-checkout --depth 1 --filter=blob:none https://github.com/philshiu/Drosophila_brain_model.git /tmp/shiu
(cd /tmp/shiu && git fetch origin d386555d1a5f40ebfa1380bcb05b1fab044855fd && git checkout HEAD -- Connectivity_783.parquet Completeness_783.csv)
mv /tmp/shiu/Connectivity_783.parquet /tmp/shiu/Completeness_783.csv data/raw/
gh api -H "Accept: application/vnd.github.raw" repos/flyconnectome/flywire_annotations/contents/supplemental_files/Supplemental_file1_neuron_annotations.tsv > data/raw/Supplemental_file1_neuron_annotations.tsv
uv run python tools/build_flywire.py     # ~30 s: data/circuits/flywire_v783.npz (125 MB)
uv run python tools/fidelity.py          # ~5 dk: data/circuits/tiers.json (arayüzdeki doğruluk/hız bilgisi)
```

(`gh api` ile 100 MB'lık dosya akışı yarıda kesiliyor, bu yüzden büyük dosya git ile alınır.
`sez_neurons.pickle` indirilmez: pickle kod çalıştırabilir, devre için gerekmiyor.)

**Lisans ve atıf.** Kod: Shiu ve ark. deposu MIT. FlyWire verisi CC-BY 4.0, atıf gerekir:
Dorkenwald ve ark. 2024 (*Nature*, FlyWire bağlantısı), Schlegel ve ark. 2024 (*Nature*, hücre
tipleri), Shiu ve ark. 2024 (*Nature*, tüm-beyin LIF modeli).

**Devre.** 138.639 nöron, 15,1 milyon sinaps (kenar), işaret × sinaps sayısı × 0,275 mV. Girdi:
imleç yaklaşınca LPLC2 (boyut) ve LC4 (hız) nöronları Poisson ateşler; bunlar Giant Fiber'a
sinaps yapar (Ache ve ark. 2019, von Reyn ve ark. 2017). Çıktı: DNp01 (Giant Fiber) → kalkış;
DNp09/P9 (Bidaye ve ark. 2020) → ileri yürüme. Hazır ama henüz kullanılmayan çıkışlar: MDN (geri
yürüme, Bidaye 2014; görsel geri çekilme, Sen 2017), DNa01/DNa02 (yürürken yönlendirme,
Rayshubskiy ve ark.). Optik lob (77,5 bin nöron, %56) simüle edilmez: görsel girdi doğrudan
projeksiyon nöronlarına verilir.

**Katmanlar.** Modelde kendiliğinden aktivite yoktur: hiç ateşlemeyen bir nöron diğerlerini
etkilemez, atılması hiçbir şeyi değiştirmez. Nöronlar, tam beyin simülasyonunda looming girdisi
altında ne kadar ateşlediklerine göre sıralanır (girdi ve çıktı nöronları hep ilk sırada); N'inci
katman ilk N nörondur. Ölçülenler (i5-10300H, tek çekirdek, dt 0,5 ms, eğitimde olmayan girdi
düzeyleri; "en kötü" = en güçlü looming):

| Nöron | GF hatası | DN korelasyonu | DN ateşlemesi korunan | Hız (ort. / en kötü) |
|---:|---:|---:|---:|---:|
| 500 | %13 | 0,996 | %75 | ×21 / ×9 |
| 1.000 | %6,4 | 0,999 | %99 | ×15 / ×8 |
| **2.000** | **%2,3** | **1,000** | **%99,9** | **×15 / ×4,3** |
| 5.000 | %2,1 | 1,000 | %99,9 | ×7,8 / ×3,0 |
| 20.000 | %0,2 | 1,000 | %100 | ×3,9 / ×1,3 |
| 50.000 | %0,1 | 1,000 | %100 | ×2,9 / ×0,8 |
| 138.639 (tam) | 0 | 1,000 | %100 | ×1,4 / ×0,2 |

Sınır: bu doğruluk **yalnız looming girdisi için** ölçüldü. Başka bir uyaran (ör. küçük nesne
nöronları) eklenirse sıralama o uyaranla yeniden kurulmalı (`tools/build_flywire.py`).

## Motor (`neuropest/engine`)

- Nöron modeli: Shiu ve ark. 2024 (`philshiu/Drosophila_brain_model`) ile aynı LIF sabitleri ve
  eşitlikler, adım başına tam (üstel) entegrasyon, 1,8 ms sinaptik gecikme.
- `LIFEngine`: numba, olay tabanlı. Her adımda yalnız **aktif** nöronlar güncellenir; maliyet ≈ aktif
  nöron × adım sayısı + sinaptik olaylar, toplam nöron sayısı değil.
- `ReferenceEngine`: yoğun NumPy sürümü, test kâhini (`tests/test_engine.py` ikisini karşılaştırır).
- Motor ayrı süreçte çalışır (`runner.py`), arayüz yük altında donmaz; gerçek zamana göre hızını
  ayarlar, yetişemezse "yavaş çekim" uyarısı çıkar.
- Araçlar (`tools/`): `bench_engine.py`, `bench_brain.py` (hız), `probe_circuit.py` (looming → DN
  yanıtı), `fidelity.py` (katman doğruluğu), `calibrate.py` (davranış ayarı), `inspect_flywire.py`,
  `explore_types.py` (hücre tipi arama), `startup_time.py`.

Az boş bellekli makinelerde (sayfa dosyası yoksa) numpy/BLAS iş parçacıkları başlarken bellek ayırmayı
başaramayabilir; paket bunu `OPENBLAS_NUM_THREADS=1` ile önler. Tam beyin katmanı ~0,4 GB ister.

## Sıradaki işler

1. Davranış eşlemesini genişletmek: MDN ile geri çekilme, DNa02 sağ-sol farkıyla yön, daha fazla
   görsel nöron tipi (her biri için sıralamayı yeniden kurmak).
2. Otomatik boyut seçimi (makineyi ölç, gerçek zamanı tutan en büyük katman).
3. Gerçek sprite'lar ve daha iyi yürüme/uçma animasyonu.
