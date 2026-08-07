# Why v7 Returns Nothing Useful for Receipt Searches

## The problem

The v7 query has no receipt field. It only searches name, A-number, and date of birth. A receipt-only search leaves every placeholder blank, so after they're stripped the query has no criteria left — which the system reads as "match everyone" and returns the same random person each time. Production works because it was built to look up receipts; v7 wasn't.

## Current v7 query (no receipt field)

```json
{
  "query": {
    "bool": {
      "minimum_should_match": 1,
      "should": [
        { "match": { "biographicInfo.name.first": "{{FIRSTNAME}}" } },
        { "match": { "biographicInfo.name.last":  "{{LASTNAME}}" } },
        { "match": { "_search.identifiers.ALIEN_NBR": "{{ANUMBER}}" } },
        { "term":  { "_search.dateOfBirth": "{{DOB}}" } }
      ]
    }
  }
}
```

For a receipt-only search, all of these are blank and get removed, leaving an empty query.

## The fix — add a receipt clause

```json
{
  "query": {
    "bool": {
      "minimum_should_match": 1,
      "should": [
        { "match": { "biographicInfo.name.first": "{{FIRSTNAME}}" } },
        { "match": { "biographicInfo.name.last":  "{{LASTNAME}}" } },
        { "match": { "_search.identifiers.ALIEN_NBR":   "{{ANUMBER}}" } },
        { "match": { "_search.identifiers.RECEIPT_NBR": "{{RECEIPT}}" } },
        { "term":  { "_search.dateOfBirth": "{{DOB}}" } }
      ]
    }
  }
}
```

## Make the exact receipt win the top spot (high boost)

```json
{ "match": { "_search.identifiers.RECEIPT_NBR": { "query": "{{RECEIPT}}", "boost": 1000000 } } }
```

## Confirm before shipping

1. **Field name** — `_search.identifiers.RECEIPT_NBR` is assumed from the A-number pattern (`_search.identifiers.ALIEN_NBR`). Verify the real field with Arvin:

   ```json
   GET /iis-identity-api-alias/_mapping/field/*RECEIPT*
   ```

2. **Placeholder** — adding `{{RECEIPT}}` means the query-building code must pass the receipt value in. The test notebook already extracts `RECEIPT`, so once the clause exists, the CRIS and UIPATH searches stop showing "(template cannot build this search)" and return real people.
