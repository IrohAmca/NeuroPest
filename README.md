# NeuroPest

Masaüstünde gezen, fare imlecine tepki veren bir sinek. Davranışı, gerçek sinek beyni
bağlantısından (FlyWire) çıkarılan bir devrenin simülasyonundan gelir (leaky integrate-and-fire).

**Durum (v0.6):** imleç sinek için üç girdiye dönüşür ve gerçek bağlantı bunları davranışa çevirir:
yaklaşma hızı → hafifse **geri yürüme** (MDN), çok hızlıysa **uçarak kaçış** (Giant Fiber); imlecin
yönü → sağ/sol **dönüş** (DNa02). Yürüme komut nöronlarına (DNp09/P9) tonik sürücü verilince yürür.
Devrenin boyutu arayüzden seçilir; her boyutun tam beyne göre doğruluğu ve hızı ölçülmüştür.
İsteğe bağlı **görsel girdi** (kontrol penceresindeki kutu) imleç sayıları yerine sineğin gözünden görüntüyü
kullanır: ekran düzlemi eğimi çok düşük bir huniyle örneklenir (aşağıda "Görsel girdi").

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
- "Görsel girdi" kutusu (kontrol penceresinde, "Ekranı sineğin gözüyle gör") açılınca imlecin sayıları
  yerine görüntü kullanılır; "Göz yüksekliği" ekranın kaç px üstünden bakıldığını ayarlar. Göz verisi
  (`tools/build_eye.py`) yoksa kutu kapalıdır.
- "Hareketlilik" yürüme sürücüsünü ayarlar: düşükse durur, ortada arada yürür, yüksekse yürür.
- Sinek, seçilen monitörün görev çubuğunun üstündeki alanda kalır. Overlay tıklamayı geçirir.
- Kontrol penceresi (koyu tema, kartlar halinde): sineğin anlık durumu, canlı ölçümler (Giant Fiber,
  MDN, yön, gerçek zaman çarpanı, CPU, aktif nöron), davranış ayarları, devre ve boyut (seçilen katmanın
  ölçülmüş doğruluğu ve hızıyla), zaman adımı, hesaplama donanımı, görünüm. Pencereyi kapatmak yalnız
  gizler; tepsi simgesine tıklayınca geri gelir, çıkış tepsi menüsünden.
- Tepsi simgesi: sineğin gözleri durumuna göre renk alır (yürüme yeşil, geri çekilme turuncu, kaçış
  kırmızı); menüde durum, pencereyi açma, sineği gizleme ve çıkış var.
- `uv run python tools/ui_preview.py` arayüzü motor çalıştırmadan PNG olarak çizer.

## Gerçek veri (FlyWire v783)

Ham dosyalar git'e girmez (`data/raw/`, `data/circuits/` yok sayılır). Kaynaklar ve git blob özetleri
(`git hash-object` ile doğrulanır):

| Dosya | Kaynak | Boyut | SHA-1 |
|---|---|---|---|
| `Connectivity_783.parquet` | github.com/philshiu/Drosophila_brain_model | 100,8 MB | `d386555d1a5f40ebfa1380bcb05b1fab044855fd` |
| `Completeness_783.csv` | aynı depo | 3,3 MB | `b5a26b82b69a3c2e7fd99cd6810c36fc8f0e492b` |
| `Supplemental_file1_neuron_annotations.tsv` | github.com/flyconnectome/flywire_annotations (`supplemental_files/`) | 31,7 MB | `02e72f6c8161d3465f77fec0edf96c5d98027a9e` |

```bash
mkdir -p data/raw
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git /tmp/shiu          # ~190 MB
git clone --depth 1 https://github.com/flyconnectome/flywire_annotations.git /tmp/fwann
cp /tmp/shiu/Connectivity_783.parquet /tmp/shiu/Completeness_783.csv data/raw/
cp /tmp/fwann/supplemental_files/Supplemental_file1_neuron_annotations.tsv data/raw/
git hash-object data/raw/*               # tablodaki SHA-1'lerle aynı olmalı
uv run python tools/build_flywire.py     # ~30 s: data/circuits/flywire_v783.npz (125 MB) + data/circuits/tiers/ (katman başına küçük dosya)
uv run python tools/build_eye.py         # göz verisi: data/circuits/eye.npz ve field.npz (görsel girdi için)
uv run python tools/fidelity.py          # ~5 dk: data/circuits/tiers.json (arayüzdeki doğruluk/hız bilgisi)
```

Düz sığ klon yeterli: dosyalar bu depolarda LFS değil, normal git nesnesi. Eski tarif `git fetch origin <blob özeti>`
kullanıyordu; GitHub bir blob'u özetiyle fetch etmeye izin vermez ("bad revision"), bu yüzden kaldırıldı. `gh api` ile
100 MB'lık dosya akışı yarıda kesiliyor. `sez_neurons.pickle` indirilmez: pickle kod çalıştırabilir, devre için gerekmiyor.

**`tiers.json` makineye özeldir.** Hata sütunları (kalkış, geri yürüme, yön, DN korelasyonu) devre ve sürücü
için geçerlidir. "Hız" sütunları ise `tools/fidelity.py`'nin koştuğu makinedeki tek çekirdek ölçümüdür ve ölçüm anındaki yüke göre
oynar; başka bir CPU'da yeniden koşturun. Dosya hangi makinede üretildiğini `machine` alanında taşır ve arayüz ipucu
metninde gösterir. Hız ayrıca sürücüye bağlıdır: bu tablo tekdüze grup sürücüsüyle ölçüldü, görüntü yolu
nöronları tek tek 150 Hz'e kadar sürer ve en kötü durumda daha yavaş koşar.

**Lisans ve atıf.** Kod: Shiu ve ark. deposu MIT. **FlyWire verisi CC BY-NC 4.0** (atıf gerekir, ticari
kullanım yok): flywire.ai/guidelines "FlyWire's public release data is made available under license CC BY-NC 4.0"
diyor (2026-10-03'te doğrulandı). Makaleler CC BY 4.0'dır ama veri o lisansta değildir; önceki sürümde bu yanlış
yazılmıştı. Bu proje yalnız ticari olmayan kullanım için dağıtılabilir; ticari bir kullanım FlyWire'dan izin ister.
Atıf: Dorkenwald ve ark. 2024 (*Nature*, FlyWire bağlantısı), Schlegel ve ark. 2024 (*Nature*, hücre tipleri),
Shiu ve ark. 2024 (*Nature*, tüm-beyin LIF modeli).

**Devre.** 138.639 nöron, 15,1 milyon sinaps (kenar), işaret × sinaps sayısı × 0,275 mV. İmleç üç
girdi grubuna çevrilir (Poisson ateşleme), çıktılar descending nöronlardan okunur:

| İmleç | Girdi nöronları | Çıktı | Davranış |
|---|---|---|---|
| yaklaşma hızı (genişleme) | LPLC2 (boyut) + LC4 (hız) | DNp01 Giant Fiber | uçarak kaçış (Ache 2019, von Reyn 2017) |
| yaklaşma hızı, hafif | LPC1 | MDN | geri yürüme (MDN işlevi: Bidaye 2014) |
| yön (sağ/sol) ve yakınlık | LC10a/c-2/d, aynı taraf | DNa02, aynı taraf | o tarafa dönüş (Rayshubskiy ve ark.) |
| "Hareketlilik" kaydırıcısı | tonik akım | DNp09 / P9 | ileri yürüme (Bidaye 2020) |
| imleç sineğin üstünde (hover), dokunulan taraf | kafa kıl duyu nöronları (BM_*, taramalar hariç) + Johnston organı C/E, aynı taraf | DNg62 (aDN1) ve DNge078 (aDN2) | kafa/anten temizlenmesi (Hampel 2015; Shiu 2024) |

Hangi girdi tipinin hangi çıkışı sürdüğü literatürden değil **modelden** okundu (`tools/probe_inputs.py`,
`tools/probe_side.py`): örneğin MDN'yi en güçlü LPC1 sürüyor, literatürdeki LC16 sürmüyor; LC10'lar
aynı taraftaki DNa02'yi sürüyor. LPC1 girdisi Giant Fiber'ı baskılıyor (`tools/probe_combo.py`), bu
yüzden geri çekilme girdisi 20 Hz'de doyuyor ve kalkış yalnız yaklaşma çok hızlıysa kazanıyor: geri
yürüme ~2 /s genişlemeden, kalkış ~7 /s'den başlıyor. Eşikler ve kazançlar benim tasarım seçimim
(`neuropest/brain.py`). Kalkış bir hız değil **olay** olarak okunur: gerçek sinekte Giant Fiber'in bir iki spike'ı
kalkışı başlatır, bu yüzden 10 ms'lik pencerede iki GF spike'ı (`BrainSpec.gf_event_spikes`) hemen FLY yapar; 80 ms
üstel ortalamalı hız yolu ikinci yol ve çıkış koşulu olarak durur. Modelde ölçüm (`tools/latency.py`, 15k, 8 tohum):
kalkış 20 /s genişlemede 16 yerine 8 ms, 10 /s'de 22 yerine 14 ms, 7 /s'de 24 yerine 20 ms sürüyor; 5 /s'de medyan değişmiyor (44 ms,
ilk spike 18 ms). Kalkış eşiği ~5 /s'den ~4 /s genişlemeye iniyor (4 /s'de 12 tohumun 9'unda uçuyor). Beyin içi gecikmenin büyük kısmı zaten spike'ın kendisi; GUI yoklaması ve kare hızı ayrı kalemler. Optik lob (77,5 bin nöron, %56) simüle edilmez: görsel girdi doğrudan
projeksiyon nöronlarına verilir.

**Katmanlar.** Modelde kendiliğinden aktivite yoktur: hiç ateşlemeyen bir nöron diğerlerini
etkilemez, atılması hiçbir şeyi değiştirmez. Nöronlar, tam beyin simülasyonunda üç girdi ailesi
(ve karışımları) altında ne kadar ateşlediklerine göre sıralanır (girdi ve çıktı nöronları hep ilk
sırada); N'inci katman ilk N nörondur. Ölçülenler (i5-10300H, tek çekirdek, dt 0,5 ms, sıralamada
kullanılmayan girdi düzeyleri; "en kötü" = en güçlü uyaran). Sapma = tam beyne göre anchor nöronun
ortalama göreli hatası:

| Nöron | Kalkış (GF) | Geri yürüme (MDN) | Yön (DNa02) | DN korelasyonu | Hız (ort. / en kötü) |
|---:|---:|---:|---:|---:|---:|
| 2.000 | %29 | %75 | %98 | 0,743 | ×65 / ×28 |
| 5.000 | %13 | %20 | %23 | 0,993 | ×27 / ×7,0 |
| 10.000 | %9 | %6 | %10 | 0,997 | ×20 / ×4,4 |
| **15.000** (varsayılan) | **%0,6** | **%5,4** | **%0,8** | **0,999** | **×11 / ×2,1** |
| 20.000 | %0,1 | %0,5 | %1,2 | 1,000 | ×5,7 / ×1,7 |
| 50.000 | %0 | %0 | %0,9 | 0,999 | ×4,0 / ×0,5 |
| 138.639 (tam) | 0 | 0 | 0 | 1,000 | ×2,5 / ×0,4 |

Tablo, dokunma girdisi eklenip sıralama yeniden kurulduktan sonra yeniden ölçüldü (5.000'in geri yürüme hatası %7 → %20,
15.000'inki %1,6 → %5,4); hız sütunu ölçüm anındaki makine yüküne göre oynuyor.

Yalnız looming ile ölçüldüğünde 2.000 nöron %99,9 doğruydu; geri yürüme ve yön eklenince aktif
devre büyüdü ve ~15.000 nöron gerekti. **Sınır:** doğruluk yalnız bu üç girdi ailesi için ölçüldü;
yeni bir uyaran eklenirse sıralama onunla yeniden kurulmalı (`flywire.train_protocols`,
`tools/build_flywire.py`, `tools/fidelity.py`). Uyaran yokken CPU ≈ 0; yük yalnız imleç yakınken artar.

**Dokunma.** Overlay tıklamayı geçirdiği için "dokunma" imlecin sineğin merkezine `14 px × boyut`
yaklaşmasıdır. Beyin bağlantısında yalnız **kafanın** mekanik duyu nöronları var (gövde ve bacak kılları
ventral sinir kordonuna girer, o veri bu bağlantıda yok), bu yüzden dokunma = kafa dokunması. Model
(`tools/probe_touch.py`, tam beyin): kafa kılları + JO-C/E tek tarafta 100 Hz'de aDN1/aDN2'yi 25-40 Hz
sürer (50 Hz'de ~5 Hz, 25 Hz'de ~0); kalkış (GF) ve geri yürüme (MDN) **hiç** sürülmez. Dokunma
orta şiddetli yaklaşmada GF'yi bastırır (looming 20 Hz: GF 59 → 3 Hz), çok hızlı yaklaşmada bastıramaz
(50 Hz: 164 → 96 Hz). Sinek `groom` durumuna girer: durur, ön bacaklar kafaya sürülür.

## Görsel girdi (huni görüşü)

Kutu açıkken sinek, ekran düzleminin `h` px üstünde duruyormuş gibi görüntüyü görür. Her medulla sütunu
gövde çerçevesinde bir yöne (azimut, yükseklik) bakar; ufkun altına bakan bir yön ekranı `h / tan(-yükseklik)`
uzaklıkta keser. Yakın zemini ince alt ommatidialar, uzak ekranı ufkun hemen altındaki dar bir bant görür:
eğimi çok düşük bir huni. Sahne şimdilik imleç (30 px yarıçaplı koyu disk, nötr zemin); gerçek ekran yakalama
sıradaki iş.

```bash
uv run python tools/build_eye.py      # ham veriyle bir kez: data/circuits/eye.npz ve field.npz (+ eye_map.png)
uv run python tools/vision_runner.py  # gerçek işçi süreci, betikli imleç hareketleri: yavaş/orta/hızlı yaklaşma...
```

**Sütun yönleri ve alıcı alanlar** (`neuropest/eyebuild.py`, bağlantıdan): fotoreseptör ya da lamina
konumlarına küre uydurmak yanlış/dejenere alan verdi; sütun yönleri Mi1 konumlarına küre uydurularak,
diğer sütunlu hücreler (L1-L3, Tm, TmY...) en yakın Mi1'in yönüyle, T4/T5 ve her görsel projeksiyon nöronu
presinaptik sütunlu nöronlarının sinaps ağırlıklı ortalama yönüyle bulundu (ortalamanın uzunluğu, 0..1,
alıcı alanın darlığı). Doğrulama: projeksiyon nöronlarının alıcı alanları doğru gözün doğru bölgesine
düşüyor. Veri çerçevesi gerçek bir sineğe göre sol-elli (ayna); gözler simetrik olduğundan etiketli taraflar
kullanılır. Sol gözün R1-6 kaydı veride eksik (lamina hücreleri tam).

**Olumsuz sonuç.** Ham LIF optik lob görüntüden looming ya da hareket ayırt etmiyor
(`tools/vision_experiment.py`: lamina hücreleri görüntüyle sürülüp tam beyin GPU'da koşuldu). Karanlık disk
yaklaşırken, kayarken, uzaklaşırken, ani belirirken ve ekran bütünüyle kararırken T4/T5/Mi1/LPLC2 sessiz
kaldı (0-0,2 Hz); optik lobu dinlenme potansiyelinin üstüne çeken tonik akımla (5,5 mV) ağ rastgele
coştu (270-340 bin spike/sn, Giant Fiber düz zeminde bile ~12 Hz), uyaranlar arasında fark çıkmadı.
Sebep: derecelendirilmiş (sivri uçsuz) hücrelerin ham LIF'te ayarı yok. Bu yüzden optik lob atlanır.

**Yerine iki retinotopik dedektör** (`neuropest/vision.py`, `Features`, sütun kafesinde):

- *genişleme* (LPLC2/LC4 benzeri): bir sütun nesnenin içinde ve nesnenin kenarı çevresindeki halkanın dört
  çeyreğinin hepsinde büyüyor; en zayıf çeyrek sayılır (kayan kenar ve tek yönlü değişim 0, küçülen nesne 0),
  bir gözün tamamında aynı değişim çıkarılır. Koyu ve açık nesne ayrı kanallar.
- *küçük nesne* (LC10 benzeri): merkez çevresinden koyu ya da açık.

Sahne değişimi **aynı duruştan** ölçülür (önceki karenin sahnesi şimdiki duruşta örneklenir), böylece
sineğin kendi yürüyüşü looming sayılmaz. `neuropest/visual.py` her LPLC2/LC4 (looming), LPC1 (geri çekilme)
ve LC10 (yön) nöronunun kendi alıcı alanındaki değeri zorlanmış ateşleme hızına çevirir ve `Brain.set_vision`
ile verir; bu sırada imleç sayılarının yerine geçerler. Kazançlar ve eşikler `tools/vision_calibrate.py --grid`
ile bulundu: yavaş yaklaşma (150 px/s) dur, orta (400) geri yürü, hızlı (800, 1500) kalk; kayan, uzaklaşan
ve duran disk dur. Yavaş taban çizgisi 8 s (`tau_adapt`): 2 s iken uzun süre yakında duran imleç
uzaklaşırken bıraktığı açık iz büyüyen açık nesne gibi okunuyor, sahte geri çekilme çıkıyordu. Dokunma (imleç
sineğin üstünde) mekanik olduğundan görsel girdi açıkken de imleç konumundan gelir.

**Maliyet.** Dedektör kare başına ~1,6 ms (60 Hz, numba), motorla aynı süreçte; merkezi beyin değişmez
(girdi nöronları her katmanda var). Ekran yakalama (gerçek ekran için) Qt `grabWindow` ile GUI iş parçacığına
ağır: 2560x1440'ta tam ekran 58 ms, 600x600'lük bölge 14 ms (`tools/capture_cost.py`); ayrı süreç gerekecek.

**Sınırlar.** İmleç düz bir düzlemde uzakta ince bir şerit olur, looming yalnız yakında belirgin
(göz yüksekliği ayarı ve imleç halesi bunu dengeler). Göz yükseldikçe imleç diski büyütülür, genişleme çıkışı
ve yön sürücüsü ölçeklenir: 250 px'ten yukarıda 30 px'lik disk bir sütundan küçük kalıyor ve sinek hiç
kaçmıyordu. 100 ile 250 px arasında yavaş/orta/hızlı yaklaşma aynı davranışı veriyor (15.000 nöron ve tam
beyin GPU'da); 350 px'te hızlı yaklaşma kalkış veriyor, orta hızlı yaklaşma geri çekilme vermiyor. Optik lob
simüle edilmiyor. Katman doğruluğu görüntüyle
(mekânsal olarak seyrek) yeniden ölçülmedi; senaryolar 5.000 ve 15.000 katmanda denendi (5.000'de eşiklere
daha az pay var: kalkışta Giant Fiber ~23 Hz, eşik 15). 15.000 katmanda şiddetli kaçış sırasında (yüz binlerce
spike/sn) ve makine yüklüyken bir ölçümde gerçek zamanın altına (×0,8) inildi; 5.000'de ×7,4.

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
  `probe_touch.py` (dokunma girdileri → DN yanıtı), `probe_circuit.py` (bir girdi grubu →
  DN yanıtı), `probe_inputs.py`, `probe_side.py`, `probe_combo.py` (hangi girdi hangi çıkışı sürer),
  `fidelity.py` (katman doğruluğu), `calibrate.py` (davranış ayarı), `inspect_flywire.py`,
  `explore_types.py` (hücre tipi arama), `startup_time.py`. Görsel girdi: `build_eye.py`,
  `vision_experiment.py` (optik lob deneyi), `vision_calibrate.py` (sahne → davranış, kazanç ayarı),
  `vision_runner.py` (gerçek işçi, uçtan uca), `capture_cost.py`, `explore_eye*.py`.

Az boş bellekli makinelerde (sayfa dosyası yoksa) numpy/BLAS iş parçacıkları başlarken bellek ayırmayı
başaramayabilir; paket bunu `OPENBLAS_NUM_THREADS=1` ile önler. Tam beyin katmanı CPU'da ~0,4 GB,
GPU'da ~0,6 GB (sürücü + tamponlar) ister.

## GPU (isteğe bağlı)

```bash
uv sync --extra gpu        # wgpu, 3,3 MB; NVIDIA, AMD, Intel ve Apple GPU'larında çalışır, CUDA gerekmez
uv run neuropest           # kontrol penceresinde "Hesaplama": Otomatik / CPU / GPU adları
```

Kontrol penceresi GPU'ları ayrı bir süreçte bulur (adaptör taraması ~1 s ve ~100 MB bellek ister) ve bunu yalnız
"GPU'ları ara…" seçilince ya da "Otomatik" büyük bir katman isteyince yapar. "Otomatik",
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

1. Görsel girdi: imleç yerine gerçek ekran görüntüsü (ayrı süreçte yakalama, kendi overlay'ini dışarıda
   bırakarak); sineğin çevresindeki bölge ~12-15 fps.
2. Daha fazla görsel nöron tipi ve davranış (küçük nesne LC11, yaklaşma yerine kaçınma ya da takip);
   her biri için sıralamayı yeniden kurmak.
3. Otomatik boyut/donanım seçimi (makineyi ölç, gerçek zamanı tutan en küçük maliyet).
4. Gerçek sprite'lar ve daha iyi yürüme/uçma/geri yürüme animasyonu.
