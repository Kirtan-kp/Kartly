# Deploying Kartly

This guide takes one machine (a small VPS or a spare PC) from nothing to a service served over https, with backups you have tested and a way back from a bad upgrade. It uses the files in this repository: `docker-compose.deploy.yml`, `docker-compose.proxy.yml`, `deploy/Caddyfile`, `scripts/backup.sh` and `scripts/restore.sh`. Every command is run from the repository root.

Be clear about what this is: a single-machine setup for a demo or a small pilot. It is not highly available (see "What this does not do").

## 1. What you need

- A machine with Docker and the Compose plugin (`docker compose version`), 2 GB of RAM or more, and the repository checked out.
- For https: a DNS name (say `support.example.com`) whose A record points at the machine, and ports 80 and 443 open to the internet. Caddy needs port 80 to prove you own the name.
- Model access: a Groq key (free tier) and/or Ollama on the host. Without either, tickets that need a model escalate to a human.

## 2. First deployment (http, on the machine itself)

```bash
cp .env.deploy.example .env.deploy
chmod 600 .env.deploy
openssl rand -hex 24      # run twice: one value for POSTGRES_PASSWORD, one for KARTLY_API_KEY
```

Edit `.env.deploy`: set `POSTGRES_PASSWORD`, `KARTLY_API_KEY` and `GROQ_API_KEY`. Use only letters and digits in the database password (it is placed inside a URL). Then:

```bash
export KARTLY_IMAGE_TAG=$(git rev-parse --short HEAD)    # or put it in .env.deploy
docker compose --env-file .env.deploy -f docker-compose.deploy.yml up -d --build
docker compose --env-file .env.deploy -f docker-compose.deploy.yml ps
curl http://127.0.0.1:8000/healthz
```

Startup order is enforced: Postgres is healthy, the one-off `migrate` service succeeds, and only then do the API and worker start. If `migrate` fails, nothing else starts; read its log with `docker compose ... logs migrate`.

Create the first administrator (it asks for a password of at least 12 characters), and optionally load demo data:

```bash
docker compose --env-file .env.deploy -f docker-compose.deploy.yml exec api python -m kartly.reviewers add maya --role admin
docker compose --env-file .env.deploy -f docker-compose.deploy.yml exec api python -m kartly.store.seed   # demo data; never add --reset to data you want to keep
```

Open `http://127.0.0.1:8000/admin` and sign in.

To avoid typing the long prefix, define a shell alias: `alias kdc='docker compose --env-file .env.deploy -f docker-compose.deploy.yml'`. If you add the proxy (next section), include `-f docker-compose.proxy.yml` in it.

## 3. Serving it over https

Add `KARTLY_DOMAIN=support.example.com` to `.env.deploy`, then start with the proxy overlay:

```bash
docker compose --env-file .env.deploy -f docker-compose.deploy.yml -f docker-compose.proxy.yml up -d
```

Caddy gets a certificate for the name and renews it by itself; the certificates live in the `caddy_data` volume, so keep that volume across upgrades. The overlay also does three things you would otherwise have to remember:

- the sign-in cookie becomes `Secure` (`KARTLY_COOKIE_SECURE=true`), so it is never sent over http;
- the interactive API documentation (`/docs`, `/redoc`, `/openapi.json`) is switched off (`KARTLY_DOCS_ENABLED=false`);
- request bodies over 64 KB are refused at the proxy, and HSTS is sent.

Only Caddy is published to the network. The API stays on `127.0.0.1:8000` and Postgres is not published at all. Check from another machine:

```bash
curl -I https://support.example.com/healthz          # 200
curl -I https://support.example.com/docs             # 404
curl -s -o /dev/null -w '%{http_code}\n' https://support.example.com/v1/tickets   # 401
```

If the certificate does not arrive, the cause is almost always DNS not yet pointing at the machine, or port 80 blocked by a cloud firewall. `docker compose ... logs caddy` says which.

Also close everything else at the firewall: allow 22 (ssh, preferably by key only), 80 and 443; deny the rest. Docker publishes ports around some host firewalls, which is why the compose files bind the API to `127.0.0.1` rather than relying on the firewall.

## 4. Secrets

- `.env.deploy` holds every secret and is ignored by git. Keep it `chmod 600`. Never put it in the image, a ticket or a chat.
- **`KARTLY_API_KEY`** is how your storefront backend calls the API. Rotate it by editing `.env.deploy`, then `up -d` (this recreates the API and worker), then updating the caller.
- **`POSTGRES_PASSWORD`** is only read when the database volume is first created. To change it later, run `ALTER USER kartly PASSWORD '...'` in the database container (`kdc exec postgres psql -U kartly -d kartly`), then update `.env.deploy` and recreate the API, worker and migrate services.
- **Reviewer passwords** are managed with `python -m kartly.reviewers passwd NAME` (or the admin page). Disable a leaver with `disable NAME`; their sessions stop working immediately.
- **`GROQ_API_KEY`**: create a new key in the Groq console, update the file, recreate the API and worker, then delete the old key.
- If a secret may have leaked, rotate it first and investigate second.

## 5. Running it: what to watch

- **Health:** `/healthz` returns 503 when the database is unreachable. Point an uptime checker (any free one) at `https://your-domain/healthz`.
- **Alerts:** set `KARTLY_ALERT_WEBHOOK_URL` to a Slack or Discord-style incoming webhook (https only) so changes in alert state reach you. `kdc exec api python -m kartly.alerts` checks now and exits 1 if anything is firing, so it also works from cron. The admin page shows the same rules.
- **Numbers:** the admin overview, `GET /v1/metrics` and `python -m kartly.costs --days 7` (inside the api container) show volume, escalations, latency and cost.
- **Logs:** `kdc logs -f api worker`. Each container keeps at most 5 files of 10 MB.
- **Stop the agent:** on the admin page, or `kdc exec api python -m kartly.controls set agent off --by maya --reason "..."`; new tickets go to a human at once. `set writes frozen` stops refunds and cancellations from running on their own. Anyone with the reviewer role can stop; only an admin turns things back on.

## 6. Backups

State lives in one place, the Postgres volume. `scripts/backup.sh` writes a compressed dump, checks that it can be read back, and keeps the newest 14:

```bash
scripts/backup.sh                    # ./backups/kartly-YYYYmmdd-HHMMSS.dump, mode 600
```

A dump that sits next to the database is lost with the machine, so copy it elsewhere every time. Run it daily and ship it off the machine. Two ways to schedule it:

```cron
# crontab -e   (daily at 02:30; adjust the path and the destination)
30 2 * * * cd /opt/kartly && scripts/backup.sh >> backups/backup.log 2>&1 && rsync -a backups/ backup-host:/srv/kartly-backups/
```

or a systemd timer (`/etc/systemd/system/kartly-backup.service` and `.timer`):

```ini
# kartly-backup.service
[Service]
Type=oneshot
WorkingDirectory=/opt/kartly
ExecStart=/opt/kartly/scripts/backup.sh
ExecStartPost=/usr/bin/rsync -a /opt/kartly/backups/ backup-host:/srv/kartly-backups/

# kartly-backup.timer
[Timer]
OnCalendar=*-*-* 02:30:00
Persistent=true
[Install]
WantedBy=timers.target
```

(`systemctl enable --now kartly-backup.timer`.) The backup holds customer data and reviewer password hashes; treat the copies like the database itself. Set `KARTLY_KEEP` to change how many local files are kept.

### Restoring

```bash
scripts/restore.sh backups/kartly-20261118-023000.dump --yes
```

This stops the API and worker, saves the current database as `backups/pre-restore-*.dump` (so a mistaken restore can itself be undone), replaces the data, runs the migrations (so an older backup comes up on the current schema), starts the services and checks they answer. If any step fails it prints `RESTORE FAILED` and says where the safety copy is.

**A backup you have not restored is a hope, not a backup.** Practise it: restore the newest dump onto a spare machine (or a second checkout with a different compose project name) once a month, sign in, and open a ticket.

What a restore does not bring back: anything that happened after the dump (tickets, approvals, audit entries), and the contents of `.env.deploy` and the Caddy certificates (back those up separately; certificates can simply be reissued).

## 7. Upgrading and rolling back

Each release gets its own image tag, so the previous one is still on the machine.

```bash
scripts/backup.sh                                    # 1. always take a backup first
cp .env.deploy .env.deploy.bak                       # 2. note the tag in use
git pull
export KARTLY_IMAGE_TAG=$(git rev-parse --short HEAD)
docker compose --env-file .env.deploy -f docker-compose.deploy.yml up -d --build    # add -f docker-compose.proxy.yml if you use it
docker compose --env-file .env.deploy -f docker-compose.deploy.yml run --rm migrate python -m kartly.migrate --check   # schema at head?
curl -f http://127.0.0.1:8000/healthz
```

The `migrate` service runs before the new API and worker start. The worker finishes the jobs it holds before stopping (up to 170 s), so tickets in flight are not cut off.

**Roll back the code** when the new release misbehaves but the schema is compatible: set `KARTLY_IMAGE_TAG` back to the previous value and run `up -d` without `--build`. Compose recreates the containers from the old image.

**Roll back the data** when a migration changed the schema, since the old code may not run on the new schema: restore the backup you took in step 1 with `scripts/restore.sh` (which runs the migrations of the *current* checkout, so check out the old release first, set its tag, then restore). You lose what happened since the backup; that is the price, and it is why migrations should be backward compatible and why you always back up first.

## 8. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `required variable ... is missing a value` | `.env.deploy` is missing a secret, or `KARTLY_DOMAIN` with the proxy overlay | Fill it in; pass `--env-file .env.deploy` |
| `migrate` exits non-zero, nothing else starts | Wrong database password, or a failed migration | `kdc logs migrate`; the password only takes effect on a fresh volume (section 4) |
| API container unhealthy | Database unreachable | `kdc ps`, `kdc logs postgres api` |
| Tickets stay `received` | The worker is not running, or the queue is stuck | `kdc ps worker`, `kdc logs worker`; check dead letters on the admin page |
| Sign-in works on http but not on https (or the reverse) | `KARTLY_COOKIE_SECURE=true` over plain http (the browser refuses the cookie) | Use the https address, or leave the proxy overlay off locally |
| No certificate | DNS or firewall | `kdc logs caddy`; check the A record and ports 80/443 |
| `host.docker.internal` unreachable (Ollama) | Ollama listens on 127.0.0.1 only | Start it with `OLLAMA_HOST=0.0.0.0` |
| Disk filling | Docker logs or old backups | Logs are capped; check `backups/` and `docker system df` |

## 9. Production checklist

- [ ] `.env.deploy` is `chmod 600` and in no repository
- [ ] https works and `/docs` returns 404
- [ ] Only ports 22, 80 and 443 are open; ssh uses keys
- [ ] The first admin exists and no default or shared accounts do
- [ ] An uptime check watches `/healthz` and an alert webhook is set
- [ ] A scheduled backup runs, copies go off the machine, and one restore has been rehearsed
- [ ] `KARTLY_IMAGE_TAG` is set per release, so you can roll back
- [ ] Writes are in `shadow` mode until the evals with a real model pass (README, "Evals")

## 10. What this does not do

- **No high availability.** One machine and one Postgres container: if either dies, the service is down until you restore. Point-in-time recovery (WAL archiving) is not set up, so you can lose up to a day of data between backups.
- **No rate limiting or per-caller authentication.** One shared API key identifies the storefront; anyone holding it can raise tickets as any customer. Keep it server-side and rotate it.
- **Dependencies and base images are not pinned by hash or lockfile**, so a rebuild can pull newer versions.
- **The worker has no healthcheck of its own**; a hung worker shows up as the "stuck queue" alert.
- **Alert thresholds are set in code**, not in configuration.
- **The compose and proxy files were checked with `docker compose config` and by tests, but the first real build and start happen on your machine (and in CI).**
