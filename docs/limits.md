# Ограничения и квоты

| Параметр                       | Значение по умолчанию |
|--------------------------------|-----------------------|
| **Макс. размер записи**        | 256 KB                |
| **Макс. запросов в минуту**    | 600                   |
| **Макс. одновременных потоков**| 10                    |

> *Замечание:* квоты можно увеличить через службу поддержки.

## Ограничения содержимого
1. Запрещены бинарные данные без Base64.
2. Максимальная глубина вложенных структур JSON — 10 уровней.
3. Значения полей `id` должны быть уникальны в пределах пространства
   вашего аккаунта.
