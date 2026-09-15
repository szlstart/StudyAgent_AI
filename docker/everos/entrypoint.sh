#!/bin/sh
set -eu

if [ ! -f "${EVEROS_ROOT}/everos.toml" ]; then
  everos init --root "${EVEROS_ROOT}"
fi

exec "$@"
