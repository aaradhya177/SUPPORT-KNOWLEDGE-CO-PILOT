# API Rate Limits

## Standard API limits

The Northstar Desk API uses rate limits to protect service availability for all customers. Standard workspaces receive six hundred requests per minute across all API keys in the workspace. Enterprise workspaces may have a higher contracted limit, but burst traffic can still be throttled when it risks degraded performance. Rate limits apply to REST endpoints, search endpoints, webhooks that trigger API callbacks, and automated integrations that use service accounts.

## Handling throttled requests

When a client exceeds the allowed request rate, the API returns HTTP 429 with a retry-after header. Integrations should pause requests until the retry window has passed and should use exponential backoff for repeated throttling. Retrying immediately can extend the throttling period and may delay normal ticket processing. Customers should avoid running full exports during business-critical hours unless the integration uses pagination and reasonable pacing.

## Reducing API usage

Customers can reduce request volume by caching ticket metadata, using webhook events instead of polling, and requesting only fields required by the integration. Bulk endpoints should be used for large synchronization jobs because they are optimized for higher throughput. Search queries should include narrow filters such as updated_after, status, or customer_id. Broad searches across all historical tickets are slower and more likely to consume the available request budget.

## Support escalation guidance

Support agents should ask for the workspace slug, approximate request volume, endpoint names, timestamps, and any request identifiers returned by the API. Temporary limit increases are reviewed by the platform team and are not guaranteed. If the customer reports throttling below the documented limit, collect logs showing response headers and timestamps so engineering can check whether multiple integrations are sharing the same workspace budget.
