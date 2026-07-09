# Notification Settings

## Notification channels

Northstar Desk can notify users by email, in-app alerts, and mobile push notifications. Each user controls their own personal notification preferences from Profile, Notifications. Workspace administrators can set default notification behavior for new users, but they cannot silently subscribe an existing user to every alert type. Email notifications are sent for assigned tickets, mentions, watcher updates, and escalations when the user has the matching preference enabled.

## Reducing notification noise

Users who receive too many messages should review assignment alerts, watcher alerts, and mention alerts separately. Watching a high-volume ticket queue can generate many updates because every public customer reply and internal note may trigger a notification. Users can mute a single ticket thread without changing global preferences. Muting stops most routine updates for that ticket, but direct mentions and urgent escalations may still notify the user.

## Missing notifications

If a user does not receive expected notifications, support agents should confirm the user's email address, notification preferences, ticket assignment, and whether the user muted the ticket. For email delivery issues, the user should check spam and quarantine folders and verify that messages from notifications@northstardesk.example are allowed by their mail system. Mobile push notifications require the mobile app to be installed, push permission to be enabled on the device, and the user to be signed in.

## Administrator guidance

Administrators can configure workspace-level defaults for new users and escalation policies, but personal preferences remain user-controlled. If an escalation policy is not notifying the expected team, verify the policy schedule, recipient group membership, and time zone. Escalation alerts are logged in the audit trail so administrators can confirm whether Northstar Desk attempted to send the notification.
