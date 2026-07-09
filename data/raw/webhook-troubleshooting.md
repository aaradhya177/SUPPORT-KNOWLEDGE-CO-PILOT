# Webhook Troubleshooting

## Webhook delivery basics

Northstar Desk webhooks send event payloads to customer-configured HTTPS endpoints. Webhooks can be triggered by ticket creation, status changes, customer replies, internal notes, assignment changes, and tag updates. Each webhook configuration has a target URL, secret, event subscriptions, and an enabled or disabled state. Disabled webhooks do not send events until an administrator re-enables them.

## Failed deliveries

Northstar Desk expects the customer endpoint to return a 2xx response quickly. If the endpoint returns a non-2xx response, times out, or cannot be reached, delivery is marked as failed. Failed deliveries are retried with backoff for a limited period. Customers should inspect their endpoint logs for the event timestamp and request identifier. A firewall, expired TLS certificate, DNS issue, or slow endpoint can prevent successful delivery.

## Signature verification

Each webhook request includes a signature header generated from the configured webhook secret. Customers should verify the signature before trusting the payload. If signature verification fails, confirm that the endpoint is using the current webhook secret and that the raw request body is used for verification. Parsing and re-serializing JSON before verification can change the byte sequence and cause a valid signature to fail.

## Support escalation guidance

Support agents should ask for the webhook ID, target URL domain, event type, timestamp, response status, and request identifier. Do not ask customers to share webhook secrets in support tickets. If webhook deliveries are delayed across multiple customers, escalate to the platform operations queue with example webhook IDs and timestamps.
