def validate_mapping_df(
    carrier,
    df_or_columns,
    show_details: bool = False
) -> bool:
    carrier_key = normalize_header(carrier)
    log(f"[INPUT] Carrier Column Keys: {carrier_key}")

    if carrier_key not in REQUIRED_SCHEMAS:
        log(f"[INPUT] Unknown carrier: '{carrier}'. No schema available.")
        return False

    if isinstance(df_or_columns, pd.DataFrame):
        if carrier_key == "manhattan life":
            observed_columns = _collapse_two_row_headers(df_or_columns)
            log(f"[INPUT] Manhattan Life raw data columns: {observed_columns}")
        else:
            observed_columns = get_columns_from_df(df_or_columns)
            log(f"[INPUT] Carrier raw data columns: {observed_columns}")
    else:
        observed_columns = list(df_or_columns)
        log(f"[INPUT] Carrier observed columns: {observed_columns}")

    observed_norm = {normalize_header(c) for c in observed_columns if str(c).strip() != ""}
    log(f"[INPUT] Carrier normalized columns: {observed_norm}")

    for idx, required_schema in enumerate(REQUIRED_SCHEMAS[carrier_key], start=1):
        required_norm = {normalize_header(c) for c in required_schema if str(c).strip() != ""}
        missing = sorted(required_norm - observed_norm)

        if not missing:
            log(f"[INPUT] Matched carrier: '{carrier}'. Mapping is good.")
            return True

        if show_details:
            log(f"[INPUT] Missing (schema option {idx}): {missing}")

    log(f"[INPUT] Mapping file does not match for carrier '{carrier}'.")
    return False