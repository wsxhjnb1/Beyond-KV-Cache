# Security policy

Security fixes are made on the default branch. No stable release series is
currently maintained.

Report suspected vulnerabilities through a
[private GitHub security advisory](https://github.com/wsxhjnb1/Beyond-KV-Cache/security/advisories/new).
Please include the affected commit and component, realistic preconditions,
minimal reproduction steps, impact, and any suggested mitigation.

Treat model files, PyTorch `.pt` files, datasets, and generated manifests as
untrusted unless their origin and digest are verified. Never load an untrusted
pickle-capable checkpoint, expose a development server directly to the public
Internet, or put credentials in commands, logs, manifests, or issue reports.
