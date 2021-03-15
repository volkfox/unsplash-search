# unsplash-search

REST µservice for semantic and reverse image search of unsplash.
Usage:

```
curl -H "Content-Type: application/json" -X POST -d '{"prompt":["The quick brown fox"], "moat_selected": ["197_859", "23_345" ],"num": 3}' http://127.0.0.1:8000/api
```
Returns:
JSON list object with unsplash.com image URLs
