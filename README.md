# NeuroPest

Masaüstünde gezen, fare imlecine tepki veren bir sinek. Davranışı, sinir devresi
simülasyonundan çıkar (leaky integrate-and-fire).

**Durum (v0.1):** oyuncak alt devre (~150 nöron). Gerçek FlyWire bağlantı verisi henüz bağlı değil.

## Çalıştırma

```bash
uv sync
uv run neuropest
```

- İmleç hızla yaklaşırsa "looming" → Giant Fiber ateşler → sinek **uçarak kaçar**.
- Yürüme sürücüsü kaydırıcısıyla sinek **yürür**, aksi halde **durur**.
- Kontrol penceresi + tepsi simgesi; overlay tıklamayı geçirir.

## Plan

1. FlyWire v783 bağlantısından (Shiu ve ark. 2024, `philshiu/Drosophila_brain_model`, kod MIT)
   küçük bir duyusal-motor alt devre çıkar; `toy_circuit.py` yerine geçsin.
2. Devre boyutu kullanıcı seçimi (küçük / orta / tam), seyrek matris + optimizasyon.
3. Gerçek sprite'lar, daha iyi yürüme/uçma animasyonu.

Veri: FlyWire (Dorkenwald ve ark. 2024; Schlegel ve ark. 2024). Veri lisansını ve atıf
koşulunu gerçek veri bağlanmadan önce doğrula.
