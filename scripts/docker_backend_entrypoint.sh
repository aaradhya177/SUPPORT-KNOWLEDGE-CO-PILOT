#!/usr/bin/env sh
set -eu

if [ -d /app/seed_data/raw ] && [ -z "$(find /app/data/raw -type f 2>/dev/null | head -n 1)" ]; then
  echo "Seeding /app/data/raw from bundled sample documents..."
  cp -R /app/seed_data/raw/. /app/data/raw/
fi

exec "$@"
