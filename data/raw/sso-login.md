# SSO Login

## Supported SSO behavior

Northstar Desk supports SAML single sign-on for business and enterprise workspaces. When SSO is enabled, users sign in through the identity provider configured by the workspace administrator. Password login can be disabled for managed users, but emergency administrator access should be kept available during rollout. SSO settings are managed from Settings, Security, Single Sign-On by administrators with security permission.

## Common login failures

If a user cannot sign in with SSO, support agents should collect the workspace slug, user email address, identity provider name, approximate timestamp, and the error message shown to the user. Common causes include an email mismatch between the identity provider and Northstar Desk, a missing group assignment, an expired certificate, or an incorrect audience or ACS URL in the identity provider configuration. The user should retry after the administrator confirms the identity provider assignment.

## SSO rollout recommendations

Administrators should test SSO with a small pilot group before requiring it for all users. During rollout, keep at least one break-glass administrator account available with a strong password and multi-factor authentication. After confirming that SSO works for managed users, the administrator can require SSO for the workspace. Changes to SSO settings are recorded in the audit trail.

## Support escalation guidance

Escalate SSO issues to the security support queue when multiple users are locked out, when certificate rotation recently occurred, or when the identity provider returns a successful authentication but Northstar Desk still rejects the login. Include screenshots, timestamps, user email addresses, and the SAML request identifier when available.
