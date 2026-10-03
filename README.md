# NeuroPest

Masaüstünde gezen, fare imlecine tepki veren bir sinek. Davranışı, gerçek sinek beyni
bağlantısından (FlyWire) çıkarılan bir devrenin simülasyonundan gelir (leaky integrate-and-fire).

**Durum (v0.4):** imleç sinek için üç girdiye dönüşür ve gerçek bağlantı bunları davranışa çevirir:
yaklaşma hızı → hafifse **geri yürüme** (MDN), çok hızlıysa **uçarak kaçış** (Giant Fiber); imlecin
yönü → sağ/sol **dönüş** (DNa02). Yürüme komut nöronlarına (DNp09/P9) tonik sürücü verilince yürür.
Devrenin boyutu arayüzden seçilir; her boyutun tam beyne göre doğruluğu ve hızı ölçülmüştür.

## Çalıştırma

```bash
uv sync
uv run neuropest
```

İlk açılışta gerçek devre yoksa oyuncak devre (146 nöron) çalışır. Gerçek devre için aşağıdaki
"Gerçek veri" bölümüne bak.

- İmleç yavaşça yaklaşırsa sinek **geri yürür**, çok hızlı yaklaşırsa **uçarak kaçar**; yakın bir imlece
  doğru **döner**. "Ürkeklik" kaydırıcısı yaklaşmaya hassasiyeti ayarlar. 700 px'ten uzak imleç yok
  sayılır (motor boşta kalır).
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

**Devre.** 138.639 nöron, 15,1 milyon sinaps (kenar), işaret × sinaps sayısı × 0,275 mV. İmleç üç
girdi grubuna çevrilir (Poisson ateşleme), çıktılar descending nöronlardan okunur:

| İmleç | Girdi nöronları | Çıktı | Davranış |
|---|---|---|---|
| yaklaşma hızı (genişleme) | LPLC2 (boyut) + LC4 (hız) | DNp01 Giant Fiber | uçarak kaçış (Ache 2019, von Reyn 2017) |
| yaklaşma hızı, hafif | LPC1 | MDN | geri yürüme (MDN işlevi: Bidaye 2014) |
| yön (sağ/sol) ve yakınlık | LC10a/c-2/d, aynı taraf | DNa02, aynı taraf | o tarafa dönüş (Rayshubskiy ve ark.) |
| "Hareketlilik" kaydırıcısı | tonik akım | DNp09 / P9 | ileri yürüme (Bidaye 2020) |

Hangi girdi tipinin hangi çıkışı sürdüğü literatürden değil **modelden** okundu (`tools/probe_inputs.py`,
`tools/probe_side.py`): örneğin MDN'yi en güçlü LPC1 sürüyor, literatürdeki LC16 sürmüyor; LC10'lar
aynı taraftaki DNa02'yi sürüyor. LPC1 girdisi Giant Fiber'ı baskılıyor (`tools/probe_combo.py`), bu
yüzden geri çekilme girdisi 20 Hz'de doyuyor ve kalkış yalnız yaklaşma çok hızlıysa kazanıyor: geri
yürüme ~2 /s genişlemeden, kalkış ~7 /s'den başlıyor. Eşikler ve kazançlar benim tasarım seçimim
(`neuropest/brain.py`). Optik lob (77,5 bin nöron, %56) simüle edilmez: görsel girdi doğrudan
projeksiyon nöronlarına verilir.

**Katmanlar.** Modelde kendiliğinden aktivite yoktur: hiç ateşlemeyen bir nöron diğerlerini
etkilemez, atılması hiçbir şeyi değiştirmez. Nöronlar, tam beyin simülasyonunda üç girdi ailesi
(ve karışımları) altında ne kadar ateşlediklerine göre sıralanır (girdi ve çıktı nöronları hep ilk
sırada); N'inci katman ilk N nörondur. Ölçülenler (i5-10300H, tek çekirdek, dt 0,5 ms, sıralamada
kullanılmayan girdi düzeyleri; "en kötü" = en güçlü uyaran). Sapma = tam beyne göre anchor nöronun
ortalama göreli hatası:

| Nöron | Kalkış (GF) | Geri yürüme (MDN) | Yön (DNa02) | DN korelasyonu | Hız (ort. / en kötü) |
|---:|---:|---:|---:|---:|---:|
| 2.000 | %16 | %66 | %27 | 0,985 | ×17 / ×5,5 |
| 5.000 | %11 | %7 | %24 | 0,989 | ×14 / ×3,4 |
| 10.000 | %7,5 | %7,5 | %11 | 0,996 | ×7,4 / ×1,4 |
| **15.000** (varsayılan) | **%0,3** | **%1,6** | **%1,7** | **0,999** | **×6,1 / ×1,7** |
| 20.000 | %0,1 | %0,5 | %0,5 | 0,998 | ×4,2 / ×1,0 |
| 50.000 | %0 | %0 | %1,1 | 0,999 | ×2,1 / ×0,4 |
| 138.639 (tam) | 0 | 0 | 0 | 1,000 | ×1,2 / ×0,14 |

Yalnız looming ile ölçüldüğünde 2.000 nöron %99,9 doğruydu; geri yürüme ve yön eklenince aktif
devre büyüdü ve ~15.000 nöron gerekti. **Sınır:** doğruluk yalnız bu üç girdi ailesi için ölçüldü;
yeni bir uyaran eklenirse sıralama onunla yeniden kurulmalı (`flywire.train_protocols`,
`tools/build_flywire.py`, `tools/fidelity.py`). Uyaran yokken CPU ≈ 0; yük yalnız imleç yakınken artar.

## Motor (`neuropest/engine`)

- Nöron modeli: Shiu ve ark. 2024 (`philshiu/Drosophila_brain_model`) ile aynı LIF sabitleri ve
  eşitlikler, adım başına tam (üstel) entegrasyon, 1,8 ms sinaptik gecikme.
- `LIFEngine`: numba, olay tabanlı. Her adımda yalnız **aktif** nöronlar güncellenir; maliyet ≈ aktif
  nöron × adım sayısı + sinaptik olaylar, toplam nöron sayısı değil.
- `ReferenceEngine`: yoğun NumPy sürümü, test kâhini (`tests/test_engine.py` ikisini karşılaştırır).
- `WGPUEngine` (`lif_wgpu.py`): aynı model GPU'da, WebGPU (wgpu) ile; aşağıdaki "GPU" bölümüne bak.
- Motor ayrı süreçte çalışır (`runner.py`), arayüz yük altında donmaz; gerçek zamana göre hızını
  ayarlar, yetişemezse "yavaş çekim" uyarısı çıkar.
- Araçlar (`tools/`): `bench_engine.py`, `bench_brain.py`, `bench_gpu.py` (hız), `gpu_probe.py` (hangi
  GPU'lar kullanılabilir), `probe_retina.py` (fotoreseptör sürülünce yük), `explore_retina.py`,
  `probe_circuit.py` (bir girdi grubu →
  DN yanıtı), `probe_inputs.py`, `probe_side.py`, `probe_combo.py` (hangi girdi hangi çıkışı sürer),
  `fidelity.py` (katman doğruluğu), `calibrate.py` (davranış ayarı), `inspect_flywire.py`,
  `explore_types.py` (hücre tipi arama), `startup_time.py`.

Az boş bellekli makinelerde (sayfa dosyası yoksa) numpy/BLAS iş parçacıkları başlarken bellek ayırmayı
başaramayabilir; paket bunu `OPENBLAS_NUM_THREADS=1` ile önler. Tam beyin katmanı CPU'da ~0,4 GB,
GPU'da ~0,6 GB (sürücü + tamponlar) ister.

## GPU (isteğe bağlı)

```bash
uv sync --extra gpu        # wgpu, 3,3 MB; NVIDIA, AMD, Intel ve Apple GPU'larında çalışır, CUDA gerekmez
uv run neuropest           # kontrol penceresinde "Hesaplama": Otomatik / CPU / GPU adları
```

Kontrol penceresi GPU'ları ayrı bir süreçte bulur (adaptör taraması ~100 MB bellek ister). "Otomatik",
50.000 nöron ve üstü katmanlarda GPU kullanır, daha küçüklerde CPU: olay tabanlı CPU motoru boşta ya da
hafif yükte hızlıdır, GPU her adımda tüm nöronları günceller (hız yüke bağlı değil, sabit).

Tam beyin (138.639 nöron, 15,1 milyon sinaps), dt 0,5 ms, fotoreseptörlerin (R1-8, 10.582) rastgele bir
kısmı Poisson ile sürülürken, gerçek zamana göre hız (`tools/bench_gpu.py`; CPU değerleri ölçümler
arasında makine yüküne göre oynadı):

| Yük (spike/sn) | CPU (numba) | GTX 1650 (Vulkan) | Intel UHD (Vulkan) |
|---|---:|---:|---:|
| retina %10 @20 Hz (21 bin) | ×9–11 | ×12,8 | ×3,4 |
| retina %50 @20 Hz (105 bin) | ×1,9 | ×13,7 | ×3,0 |
| retina %50 @50 Hz (265 bin) | ×1,0–1,4 | ×11,7 | ×3,1 |
| retina %100 @50 Hz (538 bin) | ×0,5 | ×9,1 | ×2,5 |
| retina %100 @100 Hz (1,1 milyon) | ×0,2–0,3 | ×7,0 | ×2,1 |
| looming 50 Hz (33 bin) | ×0,4 | ×8,6 | ×2,7 |

GPU ve CPU motorları tam beyin ölçeğinde aynı spike sayısını veriyor (ör. 537.701 ve 538 bin
spike/sn). GPU çekirdeği bellek bant genişliğine bağlı: durum diziye bölündü ve yalnız değişince
yazılıyor (hız 2× arttı); sinaps girdisi sabit noktalı (1/4096 mV) ve atomik toplanıyor, Poisson üreteci
hash tabanlı (istatistiksel olarak aynı). `n_active` GPU'da izlenmez. Gerçek çalışma döngüsünde (12 ms'lik
parçalar, gerçek zamana hızlanma bekleyerek) GTX 1650'de tam beyin ×5, işlemcinin ~%19'u; Intel UHD'de
~×1,3. Python tarafı adım başına ~14 µs (tek dağıtım).

## Sıradaki işler

1. Görsel girdi: ekranı "huni" geometrisiyle (bakış açısı → ekran düzlemi) ommatidia örneklerine çevirmek,
   fotoreseptörleri sürmek; optik lobu LIF ile mi, derecelendirilmiş (rate) modelle mi, yoksa projeksiyon
   nöronlarını retinotopik doğrudan sürerek mi simüle edeceğimizi deneyle seçmek.
2. Daha fazla görsel nöron tipi ve davranış (küçük nesne LC11, yaklaşma yerine kaçınma ya da takip);
   her biri için sıralamayı yeniden kurmak.
3. Otomatik boyut/donanım seçimi (makineyi ölç, gerçek zamanı tutan en küçük maliyet).
4. Gerçek sprite'lar ve daha iyi yürüme/uçma/geri yürüme animasyonu.
