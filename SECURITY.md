# Security Policy

## Supported Versions

Only the latest commit on `main` receives fixes.

## Reporting a Vulnerability

Do not open a public issue. Use GitHub's [private vulnerability reporting](https://github.com/Charis-E-Shiny/LumaForge/security/advisories/new) if it is available, or contact the repository owner through [GitHub](https://github.com/Charis-E-Shiny).

Please include the affected component, steps to reproduce, and the likely impact. This is an academic project, so responses are best effort.

## Security Design

| Area | Approach |
|---|---|
| Secrets | `MONGODB_URI` lives in environment variables and is never committed |
| Uploads | Decoded in memory, limited in size, and never stored |
| Database | Stores run metadata only, never images |
| Sample lookup | IDs are validated, which prevents path traversal |
| Availability | Short database timeouts, so a database failure cannot stop image processing |

## Known Limitations

- The public API has no authentication or rate limiting of its own, so anyone with the URL can use compute and write telemetry rows.
- The upload decoder relies on Pillow, so keep dependencies up to date.

## For Deployers

- Use a dedicated MongoDB user with access to the `lumaforge` database only.
- Rotate the connection string if it is ever exposed.
