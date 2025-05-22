# Быстрый старт с API

## 1. Получите API‑ключ
1. Перейдите в личный кабинет.
2. Откройте раздел **Настройки → API**.
3. Нажмите **Сгенерировать ключ**. Сохраните его — повторно он не отображается.

## 2. Первый запрос
```bash
curl -H "Authorization: Bearer <API_KEY>" \
     -H "Content-Type: application/json" \
     -d '{"record": {"id": "42", "payload": "Hello!"}}' \
     https://api.productx.io/v1/records
```
Ожидаемый ответ `201 Created`:
```json
{
  "status": "success",
  "record_id": "42"
}
```

## 3. Проверка статуса
```bash
curl -H "Authorization: Bearer <API_KEY>"          https://api.productx.io/v1/records/42/status
```
