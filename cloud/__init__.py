"""Cloud market scanner for the always-on alert pipeline.

This package is the part of the dashboard that runs off-machine: a GitHub
Actions job sweeps every US equity, crypto pair, forex pair and commodity
future, detects freshly triggered technical signals and relays a consolidated
alert to the phone through the Cloudflare worker's /send endpoint.

It lives beside the Streamlit app rather than inside it because the sweep needs
a real VM (no serverless CPU ceiling) and cannot depend on the local machine
being switched on.
"""
