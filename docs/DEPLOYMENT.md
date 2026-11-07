# Deployment

The application is designed for Vercel, with MongoDB Atlas as optional storage. It is live at [lumaforge-one.vercel.app](https://lumaforge-one.vercel.app), currently without MongoDB.

## Before Deploying

Confirm that these files exist and are committed:

- `models/lumanet_compact_int8.onnx`
- `models/lumanet_learned_int8.onnx`
- `samples/` and `public/samples/`
- `results/summary.json`

## MongoDB Atlas

1. Create a free cluster.
2. Create a database user with access to the `lumaforge` database only.
3. Allow network access for Vercel. Vercel functions do not use fixed outbound addresses by default, so this usually means allowing `0.0.0.0/0` with a strong password.
4. Copy the connection string.
5. Seed the results and indexes:

```bash
MONGODB_URI="<your connection string>" python scripts/seed_mongo.py
```

## Vercel

```bash
vercel login
vercel link
vercel env add MONGODB_URI production
vercel deploy --prod
```

Pushes to `main` deploy to production automatically, and each deployment appears under Deployments on GitHub.

`vercel.json` routes `/api/*` to the Python function and bundles the models and samples.

Vercel installs from `pyproject.toml` when one exists, so this repository must not have one: lint and test settings live in `ruff.toml` and `pytest.ini`, and dependencies live in `requirements.txt`. `.python-version` pins Python 3.12, because Vercel no longer offers 3.11. Tests have run on 3.11, and the live API has been checked on 3.12.

## Verify

```bash
curl https://<your-domain>/api/health
```

Expect `"status": "ok"`, and `"database": "connected"` if MongoDB is configured. Then run a capture in the Lab, and open `/results`.

## Environment Variables

| Name | Required | Default | Purpose |
|---|---|---|---|
| `MONGODB_URI` | No | | Enables telemetry and database-backed results |
| `MONGODB_DB` | No | `lumaforge` | Database name |
| `LUMAFORGE_THREADS` | No | `2` | ONNX Runtime threads |
