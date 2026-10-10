# EMA Voice

[English](README.md) · **Türkçe**

**2.736 hazır sesle** konuşan ve **10 saniyelik bir kayıttan ses klonlayan** Türkçe metinden konuşma modeli.
48 kHz çıktı verir; NVIDIA GPU'da da dizüstü bilgisayarın işlemcisinde de çalışır.
[EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) üzerine kuruludur.

## Sonuçlar

Modelin görmediği cümlelerle ölçüldü (8 adım, CFG 2.0).

| Ölçüt | Değer |
|---|---|
| Anlaşılırlık, WER (Whisper large-v3 turbo, 200 cümle) | **%6,9**; aynı codec'ten geçen gerçek kayıtlarda %7,7 |
| Ses benzerliği, üretilen ↔ gerçek (WavLM) | **0,944**; aynı kişinin iki gerçek kaydında 0,955 (**%99**) |
| Hız, RTX 3070 | RTF 0,06, **gerçek zamanın 16 katı** |
| Hız, Apple M5 işlemci | **gerçek zamanın 3,7 katı** |

**Modelin hiç duymadığı seslerin klonlanması** (16 ses; aynı kişinin gerçek bir kaydına benzerlik):

| Seviye | Kayıt | Ekleme süresi (RTX 3070) | Benzerlik |
|---|---|---|---|
| anında | 10 sn | saniyeler | 0,915 |
| `--refine` | 10 sn | ~2 dk | 0,931 |
| `--deep` | 3-5 dk | ~20-30 dk | 0,933 (şimdilik 2 ses) |

WER'in sınırını ses codec'i belirliyor: gerçek kayıtlar codec'ten geçince %7,7 alıyor, model ise biraz daha iyi.
Test sesleri de eğitimdekiler gibi sentetik; gerçek mikrofon kayıtlarında sonuç biraz düşük çıkabilir.
Örnekleri kendin dinle: [`samples/`](samples).

## Kurulum

Linux ya da macOS. NVIDIA GPU şart değil (işlemcide de çalışır; `--deep` GPU'suz yavaştır). Yaklaşık 10 GB disk.

```bash
git lfs install                      # model dosyası (26 MB) Git LFS ile saklanıyor
git clone https://github.com/muhammederem/ema-voice-tr.git && cd ema-voice-tr
./scripts/0_setup.sh                 # Python ortamı + modellerin indirilmesi, ~10 dk
```

NVIDIA GPU yoksa her şey kendiliğinden işlemcide çalışır.

## Kullanım

### Arayüz

```bash
./scripts/6_app.sh                   # tarayıcıda http://127.0.0.1:7860 açılır
```

**Konuş** sekmesi: metni yaz, sesi seç, düğmeye bas. **Ses ekle** sekmesi: kaydını yükle ya da mikrofonla kaydet,
sese bir isim ver, seviyeyi seç. Her şey kendi bilgisayarında çalışır.

### Komut satırı

```bash
./scripts/5_say.sh "Merhaba, bu benim sesim." d1-00002 merhaba.wav   # hazır ses → merhaba.wav
./scripts/5_say.sh --list                                            # ses listesi

./scripts/4_add_voice.sh kaydim.wav erem               # anında, 10 sn'den
./scripts/4_add_voice.sh kaydim.wav erem --refine      # aynı kayıt, ~2 dk, daha benzer
./scripts/4_add_voice.sh kayitlar/ erem --deep         # 3-5 dk konuşma, ~20-30 dk, en benzer
./scripts/5_say.sh "Merhaba, bu benim sesim." erem
```

### Python

```python
from ema_voice.api import EMAVoice

tts = EMAVoice.load("models/ema-voice.pt", voices_dir="voices")
speech = tts.say("Siparişiniz yarın kargoya verilecek.", voice="erem", path="siparis.wav")
# isteğe bağlı: speed=1.1, seed=7 (aynı seed, aynı çıktı), steps=8
```

### Kayıt önerileri

- Sessiz bir oda; mikrofon ya da telefon ağzına ~20 cm uzakta.
- [`prompts/kayit_metinleri.txt`](prompts/kayit_metinleri.txt) içindeki cümleleri oku (60 cümle ≈ 5 dk).
- En iyisi her cümleyi ayrı dosyaya kaydetmek (`01.wav`, `02.wav` ...). Okuduğun cümleyi yanına `01.txt` olarak
  koyabilirsin; koymazsan `--refine` ve `--deep` için Whisper yazıya döker.
- Eklenen sesler `voices/<isim>.pt` olarak kaydedilir; model dosyası hiç değişmez.

10 saniyelik kayıt sana benzeyen bir ses verir, birebir kopya değil. Kayıt uzadıkça ve `--deep` ile benzerlik artar.

## Nasıl çalışıyor

EMA Lightning'in 5,6M parametrelik flow-matching modeli, [DACVAE](https://github.com/facebookresearch/dacvae)
codec'i (48 kHz) için ses latent'leri üretir. Her ses 64 sayılık bir vektördür. Bir referans encoder bu vektörü kısa
bir kayıttan çıkarır. `--refine` vektörü senin kaydına göre ince ayarlar; `--deep` buna ek olarak yalnızca o ses
konuşurken devreye giren küçük LoRA adaptörleri öğrenir.

## Lisans, atıf ve sorumlu kullanım

- Kod: Apache 2.0. Temel model: [EMA Lightning](https://huggingface.co/canberkkkkkk/ema-lightning) (Apache 2.0).
  Metin normalizasyonu: [normalizer-tr](https://github.com/erdemtuna/normalizer-tr).
- Sesler: **Alania Turkish Synthetic Speech by PatientDesk AI**
  (https://huggingface.co/datasets/cloud0day3/alania-synthetic-speech-tr), CC BY 4.0.
- Codec: [DACVAE](https://github.com/facebookresearch/dacvae) (kod Apache 2.0, ağırlıklar SAM License). Üretilen
  tüm sesler AudioSeal filigranı taşır.
- **Yalnızca kendi sesini ya da açık izni olan birinin sesini ekle.** Dinleyene sesin yapay zekâ ile üretildiğini
  söyle. Gerçek bir kişiyi taklit etmek, birini kandırmak ya da dolandırmak veya birine söylemediği bir şeyi
  söyletmek için kullanma.
