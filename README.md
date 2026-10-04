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
- Sinek varsayılan olarak bağlı tüm monitörler arasında görev çubuklarının üstünde kesintisiz ve bağımsız gezer (farklı DPI ve çözünürlüklerdeki ekran sınırları portallarla bağlanır); istenirse kontrol penceresinden ("Görünüm" sekmesi) tek bir monitöre sabitlenebilir. Overlay tıklamayı geçirir.
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
# isteğe bağlı: tools/fidelity.py 5000 15000 30000 --dt 0.1 0.5 1.0 | --image | --mirror (tablolar data/probes/)
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
| yaklaşma hızı, hafif | LPC1 (**işlevsel yer tutucu**, aşağıya bak) | MDN | geri yürüme (MDN işlevi: Bidaye 2014) |
| yön (sağ/sol) ve yakınlık | LC10a/c-2/d, aynı taraf | DNa02, aynı taraf | o tarafa dönüş (Rayshubskiy ve ark.) |
| "Hareketlilik" kaydırıcısı | tonik akım | DNp09 / P9 | ileri yürüme (Bidaye 2020) |
| imleç sineğin üstünde (hover), dokunulan taraf | kafa kıl duyu nöronları (BM_*, taramalar hariç) + Johnston organı C/E, aynı taraf | DNg62 (aDN1) ve DNge078 (aDN2) | kafa/anten temizlenmesi (Hampel 2015; Shiu 2024) |

Hangi girdi tipinin hangi çıkışı sürdüğü literatürden değil **modelden** okundu (`tools/probe_inputs.py`,
`tools/probe_side.py`): LC10'lar aynı taraftaki DNa02'yi sürüyor. LPC1 girdisi Giant Fiber'ı baskılıyor (`tools/probe_combo.py`), bu
yüzden geri çekilme girdisi 20 Hz'de doyuyor ve kalkış yalnız yaklaşma çok hızlıysa kazanıyor: geri
yürüme ~2 /s genişlemeden, kalkış ~7 /s'den başlıyor. Eşikler ve kazançlar benim tasarım seçimim
(`neuropest/brain.py`). Kalkış bir hız değil **olay** olarak okunur: gerçek sinekte Giant Fiber'in bir iki spike'ı
kalkışı başlatır, bu yüzden 10 ms'lik pencerede iki GF spike'ı (`BrainSpec.gf_event_spikes`) hemen FLY yapar; 80 ms
üstel ortalamalı hız yolu ikinci yol ve çıkış koşulu olarak durur. Modelde ölçüm (`tools/latency.py`, 15k, 8 tohum):
kalkış 20 /s genişlemede 16 yerine 8 ms, 10 /s'de 22 yerine 14 ms, 7 /s'de 24 yerine 20 ms sürüyor; 5 /s'de medyan değişmiyor (44 ms,
ilk spike 18 ms). Kalkış eşiği ~5 /s'den ~4 /s genişlemeye iniyor (4 /s'de 12 tohumun 9'unda uçuyor). Beyin içi gecikmenin büyük kısmı zaten spike'ın kendisi; GUI yoklaması ve kare hızı ayrı kalemler. Optik lob (77,5 bin nöron, %56) simüle edilmez: görsel girdi doğrudan
projeksiyon nöronlarına verilir.

**Durumlar ve öncelik.** Altı durum: dur, yürü, uç, geri çekil, temizlen (groom), **don** (freeze). Öncelik: kalkış >
geri çekilme > donma > temizlenme > yürü/dur; üst basamak alttakini hemen keser, bir durumdan çıkış kendi histerezisini ve
asgari süresini ister. Temizlenme eskiden geri çekilmeden önce geliyordu, yani dokunulan sinek hiç geri yürümüyordu. Şimdi **kımıldayan** bir imleç (kafa üstünde yaklaşma hızı ~2 /s'yi geçince, 14 px dokunma yarıçapında
~60 px/s) geri çekilmeyi tetikler, hareketsiz hover hâlâ temizletir; dokunma GF'yi baskılamaya devam eder (kalkış yok).
**Donma bağlantıdan okunmaz, tasarlanmış bir kuraldır:** modelde sineği durduran bir nöron yok. En yakın nesnenin
genişleme hızı (1/s; görüntü yolunda dedektör çıkışı, imleç sayıları yolunda yaklaşma hızı / mesafe) 80 ms üstel
ortalamayla 0,7'yi (`FREEZE_ON`) geçerse ve geri çekilme ya da kalkış tetiklenmediyse sinek en az 1,2 sn donar, genişleme
0,3'ün altına inince çıkar. Eşik `tools/vision_calibrate.py --verbose` ile seçildi (küre modeli, 20-300 px'te aynı):
yavaş yaklaşma (150 px/s) tepe 0,89 /s → donar; kayan disk 0,49, uzaklaşan ~0, duran 0 → durur; 400 px/s geri yürür, 800 ve
1500 uçar (donma bunlarda yalnız 0-0,6 sn sürer, sonra kalkış ya da geri çekilme keser). Temas halindeki nesne yaklaşıyor sayılmaz (dokunma varken
donma girdisi 0). Literatürde yürüyen sineklerin çoğu looming'e donarak yanıt verir, az bir kısmı zıplar (Zacarias 2018);
bu kural o oranı ayarlamaz, yalnız "zayıf looming → dur" davranışını ekler. `BrainSpec.freeze_on = 0` kuralı kapatır.
Kendiliğinden durma ve kalkış (uyaransız) bu kuralın dışında, ayrı bir karar olarak açık.

**Kaçış yönü** (`tools/probe_escape.py`, `data/probes/escape_direction.csv`; tam beyin, gerçek `VisionDrive`, 800 px/s
yaklaşma, 9 yönden). Sinek şimdi kaçış yönünü imleç vektöründen (`imleç yönü + π`, `fly.py`) yazıyor; Dombrovski 2023'e göre
yön DNp02/DNp11'in tarafa bağlı LC4 gradyanıyla kodlanır. Modelde DNp02 ve DNp11 (her tipten yanda tek nöron) gerçekten
**aynı taraftaki** uyaranla ateşliyor: soldan 120°-60° yaklaşmada DNp02_L 8-13 spike, DNp02_R 0; sağdan 60°-90°'de DNp02_R
5-7, DNp02_L 0; DNp11 yalnız ön ±30° bölgesinde (L 9 / R 0 solda, 7 sağda). (R - L, yön) korelasyonu DNp02 için +0,84,
DNp11 için +0,25, Giant Fiber için +0,46. Ama bu bir **işaret** bilgisi: 1,5 sn'de 0-13 spike, kalkış kararı ise ilk 10 ms'de
verilir; tam karşıdan yaklaşmada (0°) DNp11_L 9 spike, DNp11_R 0 (soldan sapma, LC10-DNa02 asimetrisiyle aynı yönde). Bu
yüzden kaçış yönü beyinden okunacaksa en fazla "uyaranın hangi yanında" (üç sınıf) kararı için kullanılabilir ve
DNp02/DNp11'i `ANCHOR_GROUPS`'a ve sıralamaya almak gerekir (önbellek ve katmanların yeniden kurulması, doğruluk tablosunun
yeniden ölçülmesi). **Yapılmadı:** yön hâlâ imleç vektöründen; bu bir ölçüm ve karar bekleyen öneri.

**LPC1 bir işlevsel yer tutucudur.** Literatürde LPC1 bir looming dedektörü değil, T4b/T5b girdisi alan geriden-öne
translasyonel optik akış dedektörüdür; aktive edilince sinek yavaşlar ve durur, geri yürümez (Isaacson ve ark. 2023).
MDN'nin görsel girdisi LC16'dır ve bağlantı polisinaptiktir (Sen ve ark. 2017; Wu ve ark. 2016). Modelde LPC1'in MDN'yi
sürmesi bu yüzden biyolojik bir bulgu olarak okunmamalı: tekdüze Poisson sürücüsü ve sıfır bazal aktivite altında LPC1
yolunun ölçülen bir sonucu. Tam beyinde yeniden prob edildi (`tools/probe_retreat.py`, ham tablolar
`data/probes/retreat_sim.csv` ve `retreat_wiring.csv` depoda; 123 VPN tipi (≥4 nöron; tüm 326 tipin bağlantı tablosu ayrıca), 4 hız düzeyi, tümü ve yalnız ön görüş alanındaki
nöronlar):

| Tip | n | MDN 25 Hz'de | MDN 50 Hz'de | MDN 100 Hz'de | GF (en çok) | Not |
|---|---:|---:|---:|---:|---:|---|
| **LPC1** | 164 | 19 | 36,5 | 55 | 0 | ön yarı: 0 / 10 / 28 Hz |
| **LC16** | 151 | 0 | 0 | 23,5 | 0 | ön yarı (48 nöron): 100 Hz'de bile 0 |
| LC6 | 125 | 0 | 3,8 | 35 | 194 | MDN'den çok GF'yi sürüyor (kalkış girdisi) |
| LPLC1 | 140 | 0 | 0 | 0 | 12 | 25 Hz'de 10 bin nöronluk çığ var, MDN yok |
| LT82b, LC22, LPLC4, LC9 | 4-179 | 0 | 0-4 | 6-23,5 | 0 | zayıf, tekdüze sürücüde |

Okuma: (1) **LC16 gerçekten MDN'yi sürüyor** (literatür modelde de doğrulanıyor), ama eşiği yüksek: tüm 151 nöron ≥100 Hz,
LPC1 25 Hz'de başlıyor; dedektör görüntüden bu kadar yüksek oran üretmedikçe LC16 yolu geri çekilmeyi sürmez. (2) LC6 ve LPLC1
geri yürüme girdisi değil: LC6 GF'yi sürüyor, LPLC1 hiç MDN sürmüyor. (3) MDN'ye hiçbir VPN tipi tek sinapsla bağlı değil
(en güçlü tek sinaps -1,9 mV, LC33); iki sıçramalı işaretli ağırlık LPC1 için +8844 mV (sırada 2.), LC16 için -144 mV
(sırada 294.), yani LC16'nın etkisi doğrusal iki sıçramadan değil daha uzun/özyinelemeli yollardan geliyor. (4) LC10a/d iki
sıçramada MDN'yi baskılıyor (-16.000 mV civarı): dönüş girdisi geri yürümeyi sürmüyor, tutarlı. LPC1 modelde tutuldu çünkü
düşük girdide çalışan tek yol ve GF baskılaması geri çekilme/kalkış ayrımını sağlıyor; bunu bir davranış tasarımı olarak
okuyun. LC16'ya geçmek (retreat girdisi + sıralama + kalibrasyon + tiers yeniden kurma) ayrı bir karar: önerilen sürücü tüm
LC16'yı ≥100 Hz'e çıkaran, GF'yi baskılamayan bir dedektör.

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

**Sabitlenen nöronlar ve katman boyutu.** Katman boyutu N, her katmanda zorunlu olan 2.797 girdi ve çıktı nöronunu
(altı girdi grubu ve sekiz anchor grubu) **içerir**: "15.000" 12.203 sıralamayla seçilmiş nöron demektir, "5.000" yalnız 2.203.
`tiers.json` artık `pinned` ve `free` alanlarını taşıyor, arayüzün ipucu metni de gösteriyor. Sabitlenenleri sayıya
katmadan aile-başına birleşimle küçük katman kurmak (6-10 bin nöronda 15 bin doğruluğu olası) ölçülmedi.

**Görüntü sürücülü ölçüm** (`tools/fidelity.py --image`, ham tablolar `data/probes/fidelity_image*.csv`). Yukarıdaki
tablo her LPLC2/LC4/LPC1/LC10 nöronunu aynı oranda sürer; gerçek yolda yalnız alıcı alanını nesnenin kapladığı nöronlar,
zamanla değişen oranlarda ateşler (yaklaşmada 199-334 nöron, 1.001 sürülen nörondan). Aynı sahneler (duran, yaklaşan,
yandan yaklaşan, kayan, uzaklaşan imleç; küre modeli, gerçek `VisionDrive`) tam beyinde ve katmanlarda koşturuldu.
GF ve MDN: 200 ms'lik oranın tepesi, yön: sahne boyunca ortalama DNa02 sağ - sol; hata = sahne ortalaması
|katman - tam| / max(tam, 5 Hz). **Gürültü tabanı** aynı devrenin başka tohumla koşusu:

| Nöron | Kalkış (GF) | Geri yürüme (MDN) | Yön (DNa02) |
|---:|---:|---:|---:|
| tam beyin, başka tohum | %9,5 | %7,7 | %15,4 |
| 5.000 | %4,0 | %22,4 | %23,3 |
| 10.000 | %3,3 | %21,6 | %39,4 |
| **15.000** | %3,6 | %9,2 | %8,9 |
| 20.000 | %0,8 | %6,7 | %9,0 |
| 30.000 | %1,1 | %0 | %1,8 |
| 50.000 | %0 | %0 | %0 |

Okuma: grup sürücüsüyle 15.000'de %0,6 / %5,4 / %0,8 çıkan hatalar görüntü sürücüsüyle %3,6 / %9,2 / %8,9; ama bu
düzeyler tohum gürültüsünün (%9,5 / %7,7 / %15,4) içinde, yani **15.000 görüntü yolunda tam beyinden ayırt edilemiyor**.
5.000 ve 10.000'in geri yürüme ve yön hataları gürültünün belirgin üstünde (%22 / %21 ve %23 / %39): bu katmanlar
görüntüyle yalnız kalkışta güvenilir. Tek tohum ve 10 sahne; gürültü tabanı da tek karşılaştırma, kesin değil.

**Entegrasyon adımı dt.** `tools/fidelity.py --dt 0.1 0.5 1.0` (tablo `data/probes/fidelity_dt.csv`): her (katman, dt),
dt 0,1 ms'lik tam beyne göre; gürültü tabanı aynı devre ve dt 0,1 ile başka tohum (GF %11,6, MDN %3,9, DNa02 %13,7,
grooming %8,4, karışım %9,2). Tam beyinde dt 0,5: %10,3 / %4,1 / %22,7 / %12,6 / %5,2, yani yön dışında gürültü içinde;
dt 1: %11,3 / %5,8 / %20,0 / **%42,4** / %10,4, grooming açıkça bozuluyor. 15.000 katmanda dt 0,5 sonuçları da aynı
(%10,9 / %6,4 / %23,4 / %12,8 / %6,1), dt 0,1'de (referansla aynı adım ve tohum, yalnız katman etkisi) %3,1 / %5,6 / %5,9 / %2,5 / %3,1. Hız (bu makine, tam beyin): dt
0,1 ×1,6, 0,5 ×5,4, 1 ×9,6 ortalama; yani dt 1 ms hızı ~1,8 katına çıkarır ama grooming'i ve yönü bozar. 0,5 ms
korundu; 0,1 ms 3,4 kat yavaş ve bu ölçümde 0,5'ten ayırt edilebilir kazanç yok (yön hariç).

**Sol-sağ simetri (yeni bulgu).** `tools/fidelity.py --mirror` (`data/probes/steer_symmetry.csv`): tam beyinde LC10_L
ve LC10_R'yi eşit oranlarda ayrı ayrı sürünce karşı tarafın DNa02 hızı eşit çıkmıyor: 18 Hz sürücüde LC10_L → DNa02_L 83 Hz,
LC10_R → DNa02_R 20,5 Hz; 75 Hz'de 184 Hz'e karşı 76,5 Hz (DNa02 her yanda **tek nöron**, orta düzeyde sağ taraf
tek-nöron gürültüsüyle 7 Hz'e kadar iniyor). Sinek `steer = DNa02_R - DNa02_L` ile döndüğü için tam karşıdan simetrik bir
yaklaşma bile sola dönüş üretiyor: 400 px/s'de 2 s boyunca sol 41, sağ 14 spike; görüntü sürücülü sahnelerde
tam karşıdan yaklaşmada ortalama yön -10 ile -20 Hz. Sebep (annotasyondaki LC10 alt tipleri, tarafların bağlantı
sayıları, DNa02 sağ-sol bağlantı farkı) araştırılmadı; sayım gerçek, yorum değil. Çözüm kararı sizde (örneğin yön
kazancını taraf başına normalleştirmek, bu bir tasarım katmanı olur).

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
eğimi çok düşük bir huni. Bu, ekran düzlemindeki içerik (yakalanacak ekran görüntüsü) için doğru geometri.
**İmleç ise düzlemde yatan disk olarak değil, sineğin göz hizasında ona dönük koyu bir disk (küre) olarak çizilir**
(`VisionParams.cursor_model = "sphere"`, varsayılan): görüntü yarıçapı atan(r/d), ufukta, azimutu imleç yönü.
Düzlemdeki disk için genişleme hızı v·d/(d²+h²) iken aşağı kayma h·v/(d²+h²) olur (oran d/h): göz yükseldikçe
yaklaşma "çarpışma rotası" olmaktan çıkıp "altından geçen nesne"ye dönüşür ve dedektör haklı olarak susar.
Göz hizasındaki diskte genişleme yükseklikten bağımsız v/d'dir; göz yüksekliği yalnız ekran düzlemi içeriği
(ve eski düzlem-disk modeli, `visionparams.DISK_PLANE`) için anlamlıdır, kontrol penceresindeki kaydırıcı bu
yüzden imleç için etkisizdir ve pasif gösterilir. Gerçek ekran yakalama sıradaki iş.

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

**Maliyet ve Gerçek Ekran Yakalama.** Dedektör kare başına ~1,6 ms (60 Hz, numba), motorla aynı süreçte;
merkezi beyin değişmez (girdi nöronları her katmanda var). Gerçek ekran yakalama `neuropest/capture.py` ile
ayrı bir süreçte ve kilit gerektirmeyen paylaşımlı bellekle (`ctx.Array`) çalışır:
- Sineğin etrafındaki 480×480 px bölge doğrudan Windows GDI BitBlt ile yakalanır (~5 ms/kare, numba ile uint8 griye çevrilir).
- Dinamik kare hızı: Hareketli sahnede 8 fps (CPU çekirdeğinin ~%4'ü), durağan sahnede 2 fps (~%1'i).
- Çoklu monitör ve DPI: Per-monitor DPI farkındalığıyla sol monitörün negatif koordinatları (`-1920`) dahil tüm sanal masaüstünü destekler.
- Overlay izolasyonu: `SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)` (0x11) ile sineğin kendi overlay sprite'ı yakalamadan tamamen gizlenir, sinek kendini görmez.

**Sınırlar.** Kazançlar ve eşikler artık `tools/vision_calibrate.py` ile **gerçek `VisionDrive`** üzerinden
(halo ve yükseklik ölçeklemesi, `VisionParams` varsayılanları, işçinin 20 ms görüntü adımı ve 4 ms motor parçaları)
bulunuyor; aracın ilk sürümü sürücüyü kendi kazançlarıyla yeniden yazıyordu ve bulguları uygulamaya
taşınmıyordu. 15.000 nöron, çıkış = durum / ilk görüldüğü an:

| Göz yüksekliği | Model | 150 px/s | 400 px/s | 800 px/s | 1500 px/s | yandan 800 (60° sağ / 100° sol) |
|---|---|---|---|---|---|---|
| 20-300 px (hepsinde aynı) | **küre (varsayılan)** | dur | geri 0,76 s | kaç 0,48 s | kaç 0,24 s | kaç 0,42 / 0,48 s |
| 20 px | düzlem diski | geri (2,4 s) | kaç (0,90 s) | kaç 0,46 s | kaç 0,26 s | kaç 0,44 / 0,46 s |
| 100 px | düzlem diski | dur | geri 0,86 s | kaç 0,44 s | kaç 0,26 s | kaç 0,50 s / dur |
| 200 px | düzlem diski | dur | geri 0,76 s | kaç 0,48 s | kaç 0,24 s | geri 0,52 s / dur |
| 300 px | düzlem diski | dur | geri 0,64 s | geri 0,34 s | kaç 0,18 s | geri 0,46 s / dur |

**LPLC2 boyutu, LC4 hızı okur.** İlk sürümde ikisi de aynı hızı alıyordu; oysa LC4 genişleyen kenarın açısal
*hızını*, LPLC2 nesnenin açısal *boyutunu* kodlar (von Reyn 2017, Ache 2019). Şimdi (`VisionParams.loom_split`)
genişleme çıktısı LC4'e olduğu gibi gider; LPLC2'ye alıcı alanının (30° havuz) ne kadarının nesneyle kaplı olduğuyla
çarpılarak (`size_lo` 0,10 → 0, `size_hi` 0,80 → tam) gider. Küçük (15 px) hızlı bir disk için tepe hızlar LC4 112 Hz,
LPLC2 61 Hz; varsayılan 30 px diskte 800 px/s'de 141 / 125 Hz; kayan diskte 23 / 2 Hz. `gain_loom` 25 → 30 (LPLC2
kısılınca Giant Fiber'in kalkış eşiği için). Küçük disk 800 px/s'de artık kalkmıyor, geri yürüyor; büyük (60 px)
disk 400 px/s'de geri yürüyor, 800'de kalkıyor. Düzlem modeli (`DISK_PLANE`) eski davranışta (`loom_split=False`,
`gain_loom` 25). `rf_sigma_deg` (alıcı alanı ağırlığı, varsayılan 0 = düz havuz) denendi ve **kapalı bırakıldı**: 8-12°'de toplam girdi
düşüyor, 800 px/s'de kalkış kayboluyor (`gain_loom` 60'a çıksa da), yani ağırlıklı havuz için ayrı bir kalibrasyon gerekir.

Küre modelinde ayrıca duran, kayan (400 px/s, 150 px'ten), uzaklaşan ve 3 s yakında durup uzaklaşan imleç sineği
durduruyor; 250 px/s yaklaşma 1,36 s'de geri yürüme, 100 px/s yaklaşma ancak imleç 50 px'e girince (3,5 s) geri
yürüme veriyor. Düzlem modeli yükseklikle değişiyor (20 px'te 150 px/s ve 3 s sonra uzaklaşma yanlış geri çekilme
veriyor, 300 px'te 800 px/s artık kalkış değil geri yürüme veriyor) ve yandan/arkadan yaklaşmaya zayıf. Küre
modelinde kazanç farkı: geri çekilme kazancı 18 yerine 12 (göz hizasındaki disk yavaş yaklaşmada daha çok
genişliyor). Küre sineğin arkasını (±120° dışı) görmez: arkadan gelen imleç yalnızca imleç-sayısı sürücüsüyle
(görsel girdi kapalıyken) tepki verir. Optik lob simüle edilmiyor. Katman doğruluğu görüntü sürücüsüyle ayrıca ölçüldü
(Katmanlar bölümünün sonu). 15.000 katmanda şiddetli kaçış sırasında (yüz binlerce spike/sn) ve makine yüklüyken
bir ölçümde gerçek zamanın altına (×0,8) inildi; 5.000'de ×7,4.

**Yaklaşma hızı.** İmlecin yaklaşma hızı artık GUI'de değil işçide, GUI'nin her karede gönderdiği saat damgası,
imleç ve sinek konumlarından hesaplanıyor. Eski yöntem mesafe değişimini 50 ms'ye kırpılmış kare süresine bölüyordu:
200 ms'lik bir takılmadan sonra hız 4 kat büyük çıkıyor, sahte geri çekilme ya da kalkış doğuyordu. Şimdi bölen iki
örnek arasındaki gerçek süre, önceki imleç konumu sineğin **şimdiki** konumundan ölçülüyor (sineğin imlece doğru
yürümesi looming sayılmıyor; yürüyen sineğe 400 px/s'lik "kapanma" gönderen eski sayı geri çekilme çıkarıyordu,
`tests/test_runner.py`) ve 250 ms'den uzun boşluklar hız sayılmıyor. Görüntü yolunda kare periyodu 16,7 ms olarak
yazılıyordu ama 4 ms'lik parçalara yuvarlanıp 20 ms'ydi; artık 20 ms yazılı ve dedektörün zaman farkı iki imleç
örneği arasındaki gerçek süre.

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

1. **[Tamamlandı]** Görsel girdi: Gerçek ekran görüntüsü (`capture.py`, GDI BitBlt, ayrı süreç, paylaşımlı bellek,
   WDA_EXCLUDEFROMCAPTURE ile overlay izolasyonu, 8 fps dinamik / 2 fps durağan, çoklu monitör & per-monitor DPI).
2. Daha fazla görsel nöron tipi ve davranış (küçük nesne LC11, yaklaşma yerine kaçınma ya da takip);
   her biri için sıralamayı yeniden kurmak.
3. Otomatik boyut/donanım seçimi (makineyi ölç, gerçek zamanı tutan en küçük maliyet).
4. Gerçek sprite'lar ve daha iyi yürüme/uçma/geri yürüme animasyonu.
5. **Feromon Simülasyonu ve Kemotaksis (Tropotaxis):** İmlece ve ekran sınırlarına tanımlanabilir kimyasal alanlar
   (itici/çekici cVA ve agregasyon profilleri); sineğin kafa açısına göre iki anten arasındaki konsantrasyon farkı
   ($\Delta C = C_R - C_L$) ile yön (DNa02) ve yürüme (DNp09/GF) sürüşü; etki alanı ve şiddeti konfigürasyonu.
6. **[İlk adım tamamlandı: `mushroom.py`]** **Mushroom Body & Pekiştirmeli Öğrenme (RL / Sinaptik Plastisite):** Kenyon hücreleri $\to$ MBON sinapslarında
   dopaminerjik (PAM ödül / PPL1 ceza) 3 faktörlü sinaptik plastisite kuralı; imleç veya görsel desenlerle
   ödüllendirilen/cezalandırılan davranışların deneyimle öğrenilmesi (klasik koşullanma ve hafıza).
   Şu an: tasarlanmış (bağlantıdan okunmayan) hız modeli; rastgele PN→KC (%5 seyrek), KC→MBON (yaklaşma/kaçınma),
   PAM (besleme) / PPL1 (GF-MDN kaynaklı tehdit) ile kapılı depresyon; çıktı değerlik (−1 korku … +1 arzu) yürüme
   eğilimini, ürkekliği ve DNa02 yönünü ölçekler (korkuda yön ters döner). Saf sinek değerliği 0'dır, davranış değişmez.
   Sonra: gerçek FlyWire KC/MBON/DAN hücreleri, hata (RPE) sinyali, söndürme, belleğin diske yazılması ve arayüzü.
7. **Canlı 3D Beyin Görselleştirici (Hafif GPU/Nokta Bulutu):** Nöron soma koordinatlarının 3D nokta bulutu ve
   arka planda şeffaf nöropil kabuğu; simülasyon sürecinden gelen spike listesiyle parıldayan aksiyon potansiyeli
   dalgaları; serbest kamera (orbit), sekme gizliyken sıfır ek GPU/CPU yükü.
8. **Kanonik vs. Eğitilmiş Sinek Karşılaştırma Arayüzü (Connectome Diff & Hafıza):** Orijinal (naive) FlyWire v783
   referans ağırlıkları ile öğrenme sonrası plastisiteye uğramış ağırlıkların fark matrisi ($\Delta W = W_{öğrenilmiş} - W_{kanonik}$);
   davranışsal sapma grafiği (aynı uyaran karşısında bazal vs. eğitilmiş tepki eğrileri); sinek hafızasını profil
   olarak dışa aktarma/içe aktarma ve fabrika ayarlarına sıfırlama (amnesia).
