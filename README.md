# unsplash-search

REST µservice for semantic and reverse image search of unsplash. Requires:

1. t2.large AWS instance, 200GB storage (no CUDA)
2. moat.com features dataset in directory moat-dataset/ available here: https://moat-dataset.s3-us-west-2.amazonaws.com/features.npy
3. unsplash.com features dataset in directory unsplash-dataset/ available here: https://unsplash-dataset.s3-us-west-2.amazonaws.com/features.npy

Usage:

```
curl -H "Content-Type: application/json" -X POST -d '{"prompt":["The quick brown fox"], "moat_selected": ["197_859", "23_345" ],"num": 3}' http://127.0.0.1:8000/api
```
Returns:
JSON list object with unsplash.com image URLs
