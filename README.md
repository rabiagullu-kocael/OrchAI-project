# OrchAI — LLM Orchestration Backend

> Gelen kullanıcı isteklerini analiz ederek üç farklı akışa (Direct / RAG / Web) yönlendiren, MongoDB destekli hafıza yönetimi ve token/maliyet takibi içeren FastAPI tabanlı LLM orkestrasyon sistemi kurgulanmıştır.

---

## İçindekiler

- [Proje Amacı](#proje-amacı)
- [Mimari](#mimari)
- [Özellikler](#özellikler)
- [Klasör Yapısı](#klasör-yapısı)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [API Endpoint'leri](#api-endpointleri)
- [10 Senaryo Test Çıktısı](#10-senaryo-test-çıktısı)


---

## Proje Amacı

OrchAI; bir Python backend'i içinde LLM'in **nerede ve nasıl kullanılacağını** somutlaştıran bir PoC (Proof of Concept) sistemidir. Temel hedefler:

- `route → service → repo` katmanları arasında LLM entegrasyon noktalarını netleştirmek
- Farklı istek türleri için **akıllı routing** sağlamak
- **Kısa + uzun vadeli hafıza** ile konuşma bağlamını MongoDB'de kalıcı hale getirmek
- Her istekte **token/maliyet logu** tutarak gözlemlenebilirlik sağlamak
- Gerçek API'lerle çalışan, Swagger üzerinden test edilebilir canlı bir sistem sunmak

---

## Mimari

```
Kullanıcı İsteği
      │
      ▼
┌─────────────┐
│  FastAPI    │  /api/v1/chat
│  Route      │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  LLM            │  routing/router_engine.py
│  Orchestrator   │  → Keyword analizi
│                 │  → 3 akış kararı
└────┬──────┬─────┘
     │      │      │
     ▼      ▼      ▼
 DIRECT    RAG    WEB
   │        │      │
   │    MongoDB  DuckDuckGo
   │    Doküman   Arama
   │    Arama      │
   │        │      │
   ▼        ▼      ▼
OpenAI  OpenAI  Claude
GPT-4o  GPT-4o  Sonnet
 mini    mini
     │      │      │
     └──────┴──────┘
              │
              ▼
       ┌─────────────┐
       │  Memory     │  messages + memory_store
       │  Service    │  (short-term + long-term)
       └─────────────┘
              │
              ▼
       ┌─────────────┐
       │  Trace      │  trace_logs koleksiyonu
       │  Service    │  token / cost / latency
       └─────────────┘
              │
              ▼
         Response
```

### MongoDB Koleksiyonları

| Koleksiyon | İçerik |
|---|---|
| `sessions` | Oturum metadata, oluşturulma zamanı |
| `messages` | Her mesaj (role, content, created_at) |
| `memory_store` | Uzun vadeli özetler (long_term) |
| `documents` | RAG için kaynak dokümanlar |
| `trace_logs` | Token, maliyet, latency, route logları |

Not: RAG için documents koleksiyonunda içerikler manuel olarak eklenmiştir.JSON formatında hazır mesaj ve yanıtlar verilmiştir.

---

## Özellikler

### 🔀 3 Akışlı Routing

| Akış | Tetikleyici | Model |
|---|---|---|
| **Direct** | Selamlama, matematik, basit soru | GPT-4o-mini |
| **RAG** | Teknik/kavramsal sorular, "nedir/nasıl" | GPT-4o-mini + MongoDB |
| **Web** | Güncel bilgi, haber, "bugün/şu an" | Claude Sonnet |

<img width="1756" height="597" alt="orchai_test_cikti" src="https://github.com/user-attachments/assets/3ae627b9-e89d-4bf5-90e5-223d72a7c5f5" />


Routing kararı üç aşamada verilir:
1. `force_route` parametresi (test için)
2. Keyword/pattern eşleşmesi
3. LLM classifier (belirsiz durumlarda)

### 🧠 Hafıza Yönetimi

**Kısa Vadeli (short_term):**
- Son `MAX_SHORT_TERM_MESSAGES` (varsayılan: 10) mesaj ham halde saklanır
- Her istekte context olarak LLM'e verilir
- `messages` koleksiyonundan çekilir

**Uzun Vadeli (long_term):**
- Mesaj sayısı `LONG_TERM_SUMMARY_THRESHOLD` (varsayılan: 20) değerini geçtiğinde tetiklenir
- Eski mesajlar LLM ile özetlenerek `memory_store`'a yazılır
- Yeni konuşmalarda "geçmiş bağlam" olarak system prompt'a eklenir

**Geri Çağırma Kuralları:**
1. Her istekte önce uzun vadeli özet alınır (varsa)
2. Kısa vadeli son N mesaj alınır
3. İkisi birleştirilerek LLM'e system/context olarak verilir

### 📊 Token & Maliyet Takibi

Her istekte `trace_logs` koleksiyonuna yazılan alanlar:

```json
{
  "session_id": "...",
  "route_type": "rag",
  "model_used": "gpt-4o-mini",
  "token_usage": {
    "prompt_tokens": 350,
    "completion_tokens": 180,
    "total_tokens": 530,
    "estimated_cost_usd": 0.000106
  },
  "processing_time_ms": 1240,
  "routing_reason": "RAG keyword eşleşmesi: nedir",
  "rag_doc_count": 1,
  "web_results_used": false
}
```

### 🔍 RAG Pipeline

1. MongoDB text search ile aday dokümanlar bulunur
2. OpenAI `text-embedding-3-small` ile sorgu vektörü oluşturulur
3. Cosine similarity ile dokümanlar re-rank edilir
4. Threshold (`RAG_SIMILARITY_THRESHOLD`: 0.75) altındaki sonuçlar filtrelenir
5. En alakalı 3 doküman LLM context'ine eklenir

---

## Klasör Yapısı

```
orchai/
├── main.py                          # FastAPI app, lifespan, router mount
├── .env                             # API key'ler (git'e eklenmez)
├── requirements.txt
├── add_inits.py                     # __init__.py yardımcı script
│
└── app/
    ├── api/
    │   └── routes/
    │       ├── chat.py              # /chat, /sessions, /traces endpoint'leri
    │       └── health.py            # /health endpoint'i
    │
    ├── core/
    │   └── database.py              # DB init, index oluşturma
    │
    ├── models/
    │   ├── chat_models.py           # Request/Response Pydantic modelleri
    │   └── memory_models.py         # Memory Pydantic modelleri
    │
    ├── orchestrator/
    │   └── llm_orchestrator.py      # 3 akışı koordine eden ana motor
    │
    ├── rag/
    │   ├── embedding_service.py     # OpenAI embedding + cosine similarity
    │   └── retrieval_service.py     # MongoDB text search + re-rank
    │
    ├── routing/
    │   └── router_engine.py         # Keyword → DIRECT/RAG/WEB karar motoru
    │
    ├── services/
    │   ├── chat_service.py          # Orchestrator'a köprü
    │   ├── llm_service.py           # OpenAI + Claude API yönetimi
    │   ├── memory_service.py        # Short/long term memory
    │   ├── rag_service.py           # RAG servis katmanı
    │   ├── trace_service.py         # Token/cost loglama
    │   └── web_service.py           # DuckDuckGo web araması
    │
    ├── utils/
    │   ├── config.py                # Pydantic Settings, .env okuma
    │   └── mongo_client.py          # Motor async client
    │
    └── observability/               # Gelecek: Prometheus/OpenTelemetry
```

---

## Kurulum

### Gereksinimler

- Python 3.11+
- MongoDB Atlas veya local MongoDB
- OpenAI API Key
- Anthropic (Claude) API Key

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/kullanici/orchai.git
cd orchai

# 2. Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. .env dosyasını oluştur
cp .env.example .env
# .env dosyasını düzenle (API key'lerini gir)

# 5. Başlat
uvicorn main:app --reload
```

### .env Yapısı

```env
MONGO_URI=mongodb+srv:/*****
MONGO_DB_NAME=orchai_db
OPENAI_API_KEY=sk-proj-...
CLAUDE_API_KEY=sk-ant-api03-...
```

### MongoDB Doküman Formatı (documents koleksiyonu)

```json
{
  "title": "Chatbot Nedir",
  "content": "Chatbot, kullanıcılarla doğal dil ile iletişim kurabilen...",
  "createdAt": "2026-03-11T00:00:00Z"
}
```

---

## Kullanım

Uygulama ayağa kalktıktan sonra Swagger UI'ya eriş:

```
http://localhost:8000/docs
```

### Örnek İstek

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Chatbot nedir?"}'
```

### Örnek Yanıt

```json
{
  "session_id": "session-785200bb0ca3",
  "message_id": "msg-4fd189771b",
  "answer": "Chatbot, kullanıcılarla doğal dil ile iletişim kurabilen yazılım uygulamasıdır...",
  "route_used": "rag",
  "model_used": "gpt-4o-mini",
  "token_usage": {
    "prompt_tokens": 127,
    "completion_tokens": 25,
    "total_tokens": 152,
    "estimated_cost_usd": 0.00003405
  },
  "rag_contexts": [
    {
      "document_id": "69b0b7b6d5167409c6cf5827",
      "title": "Chatbot Nedir",
      "score": 0.7455,
      "snippet": "Chatbot, kullanıcılarla doğal dil ile iletişim kurabilen..."
    }
  ],
  "web_results_used": false,
  "processing_time_ms": 12256
}
```

---

## API Endpoint'leri

| Method | Endpoint | Açıklama |
|---|---|---|
| POST | `/api/v1/chat` | Mesaj gönder (otomatik routing) |
| POST | `/api/v1/chat/test-scenarios` | 10 senaryo testi çalıştır |
| GET | `/api/v1/health` | Sistem sağlık durumu |
| POST | `/api/v1/sessions` | Yeni oturum oluştur |
| GET | `/api/v1/sessions/{id}` | Oturum bilgisi |
| GET | `/api/v1/sessions/{id}/history` | Konuşma geçmişi |
| DELETE | `/api/v1/sessions/{id}` | Session memory temizle |
| GET | `/api/v1/sessions/{id}/memory` | Memory istatistikleri |
| GET | `/api/v1/sessions/{id}/traces` | Token/cost logları |
| GET | `/api/v1/sessions/{id}/cost-summary` | Maliyet özeti |

---
<img width="1893" height="842" alt="orchai_test" src="https://github.com/user-attachments/assets/82d96495-5403-493e-af02-87800bc12b5c" />


<img width="1905" height="860" alt="orchai_endpoint" src="https://github.com/user-attachments/assets/c94ac0da-466f-40e5-b34c-e62cded34c46" />



## 10 Senaryo Test Çıktısı

`POST /api/v1/chat/test-scenarios` endpoint'i ile elde edilen gerçek çıktı:

| # | Girdi | Route | Model | Tokens | Maliyet |
|---|---|---|---|---|---|
| 1 | Merhaba! Nasılsın? | direct | gpt-4o-mini | 101 | $0.000030 |
| 2 | Chatbot nedir ve nasıl çalışır? | rag | gpt-4o-mini | 327 | $0.000115 |
| 3 | Yapay zeka ile ML farkı nedir? | rag | gpt-4o-mini | 479 | $0.000129 |
| 4 | Transformer modeli ne demek? | rag | gpt-4o-mini | 599 | $0.000166 |
| 5 | 2 + 2 kaç eder? | direct | gpt-4o-mini | 626 | $0.000099 |
| 6 | RAG sistemi nedir? | rag | gpt-4o-mini | 851 | $0.000190 |
| 7 | Bugünkü hava durumu nasıl? | web | claude-sonnet | 1324 | $0.006384 |
| 8 | NLP ne işe yarar? | rag | gpt-4o-mini | 990 | $0.000270 |
| 9 | Derin öğrenme nasıl çalışır? | rag | gpt-4o-mini | 1313 | $0.000403 |
| 10 | Teşekkürler, görüşürüz! | direct | gpt-4o-mini | 1167 | $0.000185 |

**Toplam:** 10/10 başarılı · 7.777 token · $0.007972

---



| Kriter | Durum |
|---|---|
| LLM entegrasyon noktası net: route/service/repo sınırları çizilmiş | ✅ |
| Routing en az 3 akışı destekliyor (direct / RAG / web) | ✅ |
| Memory: kısa + uzun vadeli ayrımı var, Mongo'da saklanıyor, geri çağırma kuralları yazılı | ✅ |
| Token/cost log'u her istekte kaydediliyor (trace_logs) | ✅ |
| PoC canlı çalışıyor, örnek 10 senaryoda sonuç üretiyor | ✅ |

---

## Teknoloji Stack'i

| Katman | Teknoloji |
|---|---|
| API Framework | FastAPI |
| Veritabanı | MongoDB (Motor async driver) |
| Küçük/Hızlı LLM | OpenAI GPT-4o-mini |
| Büyük/Güçlü LLM | Anthropic Claude Sonnet |
| Embedding | OpenAI text-embedding-3-small |
| Web Arama | DuckDuckGo Instant Answer API |
| Validation | Pydantic v2 |
| Server | Uvicorn |
