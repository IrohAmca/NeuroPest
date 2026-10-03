# NeuroPest

Masaüstünde gezen, fare imlecine tepki veren bir sinek. Davranışı, sinir devresi
simülasyonundan çıkar (leaky integrate-and-fire).

**Durum (v0.2):** küçük oyuncak devre (146 nöron) + boyut/zaman adımı seçen motor. Gerçek FlyWire
bağlantı verisi henüz bağlı değil; kontrol penceresindeki 500 – 139.000 nöronluk seçenekler
**sentetik yük** (davranışı değiştirmez), gerçek alt devre gelene kadar performans kontrollerini
denemek içindir.

## Çalıştırma

```bash
uv sync
uv run neuropest
```

- İmleç hızla yaklaşırsa "looming" → Giant Fiber ateşler → sinek **uçarak kaçar**.
- "Hareketlilik" kaydırıcısı yürüme sürücüsünü ayarlar: düşükse durur, ortada arada yürür, yüksekse yürür.
- Sinek, seçilen monitörün görev çubuğunun üstündeki alanda kalır. Overlay tıklamayı geçirir.
- Kontrol penceresi: devre boyutu, zaman adımı, gerçek zaman çarpanı, aktif nöron sayısı, CPU.

## Motor (`neuropest/engine`)

- Nöron modeli: Shiu ve ark. 2024 (`philshiu/Drosophila_brain_model`, kod MIT) ile aynı LIF sabitleri
  ve eşitlikler, adım başına tam (üstel) entegrasyon, 1,8 ms sinaptik gecikme.
- `LIFEngine`: numba, olay tabanlı. Her adımda yalnız **aktif** nöronlar güncellenir; modelde
  kendiliğinden aktivite olmadığı için sessiz nöron maliyetsizdir. Maliyet ≈ aktif nöron × adım
  sayısı + sinaptik olaylar, toplam nöron sayısı değil.
- `ReferenceEngine`: yoğun NumPy sürümü, test kâhini (`tests/test_engine.py` ikisini karşılaştırır).
- Motor ayrı süreçte çalışır (`runner.py`), arayüz yük altında donmaz. Süreç gerçek zamana göre
  hızını ayarlar; yetişemezse "yavaş çekim" uyarısı çıkar.
- Ölçüm araçları: `tools/bench_engine.py`, `tools/bench_brain.py`; davranış ayarı:
  `tools/calibrate_toy.py`.

## Plan

1. FlyWire v783 bağlantısından (veri CC-BY 4.0, atıf gerekir) imleçten girdi alan ve descending
   nöronlara çıkan alt devreyi çıkar; nöronları yol önemine göre sırala, kullanıcı ilk N'i seçsin
   (iç içe katmanlar).
2. Descending nöron → davranış eşlemesini hücre tipi tablosuna (kaynaklı) taşı.
3. Otomatik boyut seçimi (makineyi ölç, gerçek zamanı tutan en büyük katman).
4. Gerçek sprite'lar ve daha iyi yürüme/uçma animasyonu.

Veri: FlyWire (Dorkenwald ve ark. 2024; Schlegel ve ark. 2024).
