# Ontology

Sources: `active_campaigns` (BigQuery), `luma_events_data` (BigQuery)

## Entity graph

```
Person ←── PARTICIPATED_BY ── Interaction ── FOR_EVENT ──→ Event
```

---

## Person

> Any individual — an AC contact or a Luma event guest.
> Full union on `email`. AC preferred when both sources have a value; Luma used as fallback.

| Attribute | Type | Source | Notes |
|---|---|---|---|
| `email` | text | `contacts.email` / `event_guests.guest__email` | Stitch key — `COALESCE(ac, luma)` |
| `first_name` | text | `contacts.first_name` / `event_guests.guest__user_first_name` | `COALESCE(ac, luma)` |
| `last_name` | text | `contacts.last_name` / `event_guests.guest__user_last_name` | `COALESCE(ac, luma)` |
| `phone` | text | `contacts.phone` / `event_guests.guest__phone_number` | `COALESCE(ac, luma)` |
| `org_name` | text | `contacts.orgname` | AC only |
| `created_at` | timestamp | `contacts.cdate` | AC only |
| `updated_at` | timestamp | `contacts.udate` | AC only |
| `ac_id` | text | `contacts.id` | AC system ID |
| `luma_user_id` | text | `event_guests.guest__user_id` | Luma system ID |

---

## Event

> Any interaction type — a Luma meetup, an AC education program (tag), or an AC mailing list.
> Luma is master for event metadata. `event_type` is derived from the source table.

| Attribute | Type | Source | Notes |
|---|---|---|---|
| `name` | text | `events.event__name` / `tags.tag` / `lists.name` | Luma master |
| `description` | text | `events.event__description` / `tags.description` | Luma master |
| `created_at` | timestamp | `events.event__created_at` / `tags.cdate` / `lists.cdate` | Luma master |
| `event_type` | text | derived | `meetup` \| `education_program` \| `list_subscription` |
| `start_at` | timestamp | `events.event__start_at` | Luma only |
| `end_at` | timestamp | `events.event__end_at` | Luma only |
| `url` | text | `events.event__url` | Luma only |
| `meeting_url` | text | `events.event__meeting_url` | Luma only |
| `location` | text | `events.event__geo_address_json__full_address` | Luma only |
| `luma_api_id` | text | `events.event__api_id` | Luma system ID |
| `ac_id` | text | `tags.id` / `lists.id` | AC system ID |

**Excluded AC tags** (not treated as events): `source1`, `Title`, `Also_Marketing_Permission`, `unsubscribed from education`, `test name field variables`, `2024`

---

## Interaction

> A person's participation in an Event — Luma guest registration, AC tag assignment, or AC list subscription.
> `interaction_type` derived from source table.

| Attribute | Type | Source | Notes |
|---|---|---|---|
| `interaction_type` | text | derived | `meetup_attendance` \| `education_enrollment` \| `list_subscription` |
| `interacted_at` | timestamp | `event_guests.guest__registered_at` / `contact_tags.cdate` / `contact_lists.sdate` | |
| `status` | text | `event_guests.guest__approval_status` / `contact_lists.status` | |
| `checked_in_at` | timestamp | `event_guests.guest__checked_in_at` | Luma only |
| `channel` | text | `contact_lists.channel` | AC lists only |

---

## Relationships

| From | Relationship | To | Via |
|---|---|---|---|
| Interaction | `PARTICIPATED_BY` | Person | `email` (all three Interaction sources link back to a Person by email) |
| Interaction | `FOR_EVENT` | Event | `event_guests._events_api_id` → `events.event__api_id`; `contact_tags.tag` → `tags.id`; `contact_lists.list` → `lists.id` |

---

## Assumptions & exclusions

1. `email` is the cross-source stitch key for Person — assumed to be populated and consistent across AC and Luma
2. AC `tags` and `lists` are both modelled as Events — tags = education programs, lists = campaign/engagement categories
3. Luma `event_guests` is a dual-role table: contributes attributes to Person and rows to Interaction
4. Nested Luma tables excluded: `event_guests__guest__event_tickets`, `event_guests__guest__registration_answers`, `events__event__registration_questions`
5. No semantic gaps — all stated use cases covered by the three entities
