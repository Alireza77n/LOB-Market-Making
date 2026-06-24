"""
Shared normalization-method definitions and the interactive selector used by BOTH data
ingestion paths:

  * data_processing/data_process.py        (LOBSTER / NASDAQ message+orderbook format)
  * data_processing/convert_exchange_data.py (Iranian-exchange combined-snapshot CSVs)

Three feature-normalization schemes are supported. None of them touch the label columns
(Raw_Target_* / Smooth_Target_* stay in raw tick units).

  global    - Single per-symbol z-score. mean/std are computed ONCE from the TRAINING
              portion only and applied unchanged to validation/test. No look-ahead.
              Best for LOW-volume data: most samples, no warm-up days dropped, but does
              not adapt to drift across a long test period.

  rolling_1 - Rolling z-score with a 1-day window: each day is normalized using the
              mean/std of the SINGLE previous day. Drops the first 1 day (no history).
              A middle ground - adapts quickly to drift but needs enough data that
              losing one day and using a 1-day estimate is acceptable.

  rolling_5 - The original LOBFrame scheme: rolling z-score with a 5-day window, i.e.
              each day normalized by the mean/std accumulated over the previous 5 days.
              Drops the first 5 days. Best for HIGH-volume data where 5 days is a stable,
              representative estimate and per-day drift matters.

`rolling_1` and `rolling_5` are the same algorithm with window=1 and window=5; the
window is exposed as ROLLING_WINDOWS so callers don't hard-code it.
"""

METHODS = ("global", "rolling_1", "rolling_5")

# Rolling window (in days) implied by each rolling method name. 'global' has no window.
ROLLING_WINDOWS = {"rolling_1": 1, "rolling_5": 5}

# Short, data-volume-oriented descriptions shown in the interactive prompt.
DESCRIPTIONS = {
    "global": (
        "Single global z-score from the TRAINING split only (no look-ahead). "
        "Best for LOW-volume data: keeps the most samples, drops no warm-up days, "
        "but does not adapt to drift over a long test period."
    ),
    "rolling_1": (
        "1-day rolling z-score: each day normalized by the previous 1 day. "
        "Drops the first day. Adapts quickly to drift; for MODERATE-volume data."
    ),
    "rolling_5": (
        "Original LOBFrame 5-day rolling z-score: each day normalized by the previous "
        "5 days. Drops the first 5 days. Best for HIGH-volume data where 5 days is a "
        "stable estimate and per-day drift matters."
    ),
}


def describe(method: str) -> str:
    return DESCRIPTIONS.get(method, "")


def normalize_method_arg(value):
    """Validate/normalize a method string passed via CLI/yaml. Returns a canonical name.

    Accepts the canonical names plus a few friendly aliases. Raises ValueError on an
    unknown value so misconfiguration fails loudly instead of silently skipping scaling.
    """
    if value is None:
        return None
    v = str(value).strip().lower()
    aliases = {
        "single": "global", "mu_sigma": "global", "musigma": "global",
        "1": "rolling_1", "rolling1": "rolling_1", "rolling-1": "rolling_1",
        "5": "rolling_5", "rolling5": "rolling_5", "rolling-5": "rolling_5",
        "original": "rolling_5", "lobframe": "rolling_5",
    }
    v = aliases.get(v, v)
    if v not in METHODS:
        raise ValueError(
            f"Unknown normalization method {value!r}. Choose one of: {', '.join(METHODS)}."
        )
    return v


def prompt_for_method(default: str = "global") -> str:
    """Interactively ask the user which normalization method to use, printing a short
    data-volume-oriented description of each. Returns a canonical method name.

    Used only when the method was not supplied via CLI flag / yaml. Falls back to
    `default` on a bare Enter or if stdin is not interactive (e.g. piped/batch runs).
    """
    import sys

    print("\nSelect a feature-normalization method for your data:")
    for i, m in enumerate(METHODS, start=1):
        marker = " (default)" if m == default else ""
        print(f"  [{i}] {m}{marker}\n      {DESCRIPTIONS[m]}")

    if not sys.stdin or not sys.stdin.isatty():
        print(f"(non-interactive stdin) -> using default: {default}\n")
        return default

    while True:
        choice = input(
            f"\nEnter 1-{len(METHODS)} or a method name [default: {default}]: "
        ).strip().lower()
        if choice == "":
            return default
        if choice.isdigit() and 1 <= int(choice) <= len(METHODS):
            return METHODS[int(choice) - 1]
        try:
            return normalize_method_arg(choice)
        except ValueError as e:
            print(f"  {e}")


def resolve_method(method, default: str = "global", interactive: bool = True) -> str:
    """Return a canonical method name: use `method` if given, else prompt (or default).

    `method` may be a canonical name, an alias, or None. If None and `interactive`,
    the user is prompted; otherwise `default` is used.
    """
    method = normalize_method_arg(method)
    if method is not None:
        return method
    if interactive:
        return prompt_for_method(default=default)
    return default


# ---------------------------------------------------------------------------
# Data-type selection
# ---------------------------------------------------------------------------
# Two ingestion sources are supported, with different normalization policy:
#   * "lobster"  : LOBSTER / NASDAQ message+orderbook files. ALWAYS uses the original
#                  framework's 5-day rolling z-score; the user is NOT asked.
#   * "exchange" : Iranian-exchange combined-snapshot CSVs. The user IS asked which of
#                  the 3 methods (global / rolling_1 / rolling_5) to use.
DATA_TYPES = ("exchange", "lobster")

DATA_TYPE_DESCRIPTIONS = {
    "exchange": "Iranian-exchange API data (Nobitex/Bitpin/Ramzinex/Tabdeal/Wallex "
                "combined-snapshot CSVs). You'll choose one of 3 normalization methods.",
    "lobster": "LOBSTER / NASDAQ message+orderbook data. Always uses the original "
               "framework 5-day rolling z-score (no further question).",
}


def prompt_for_data_type(default: str = "exchange") -> str:
    """Ask which data source is being processed. Returns a canonical data-type name."""
    import sys

    print("\nWhich data are you going to use?")
    for i, d in enumerate(DATA_TYPES, start=1):
        marker = " (default)" if d == default else ""
        print(f"  [{i}] {d}{marker}\n      {DATA_TYPE_DESCRIPTIONS[d]}")

    if not sys.stdin or not sys.stdin.isatty():
        print(f"(non-interactive stdin) -> using default: {default}\n")
        return default

    while True:
        choice = input(
            f"\nEnter 1-{len(DATA_TYPES)} or a name [default: {default}]: "
        ).strip().lower()
        if choice == "":
            return default
        if choice.isdigit() and 1 <= int(choice) <= len(DATA_TYPES):
            return DATA_TYPES[int(choice) - 1]
        if choice in DATA_TYPES:
            return choice
        print(f"  Unknown data type {choice!r}. Choose one of: {', '.join(DATA_TYPES)}.")


# ---------------------------------------------------------------------------
# Data-representation selection (raw LOB vs. Order Flow Imbalance)
# ---------------------------------------------------------------------------
# Independently of the data SOURCE and the normalization METHOD, the framework can run
# with two feature representations:
#   * "lob" : raw limit order book features only (40 columns: 4 fields x 10 levels).
#             This is the framework's historical behaviour.
#   * "ofi" : multilevel Order Flow Imbalance features (1 OFI column per level, i.e. 10
#             columns) used as the ONLY model features, REPLACING the 40 raw LOB columns.
#             OFI per level = bid order flow - ask order flow (Cont et al.). The 10 OFI
#             columns are z-scored on their own. (ASKp1/BIDp1 are retained only in the
#             unscaled files so the backtest can still reconstruct trade prices.)
DATA_REPRESENTATIONS = ("lob", "ofi")

DATA_REPRESENTATION_DESCRIPTIONS = {
    "lob": "Raw limit order book data (40 features). Original framework behaviour.",
    "ofi": "Pure multilevel Order Flow Imbalance (10 features), replacing raw LOB. "
           "OFI per level = bid order flow - ask order flow (Cont et al.). Field-standard; "
           "use with a sequence/feature model (transformer, dla, cnn1), not the LOB CNNs.",
}


def normalize_representation_arg(value):
    """Validate/normalize a data-representation string. Returns a canonical name or None."""
    if value is None:
        return None
    v = str(value).strip().lower()
    aliases = {"orderbook": "lob", "orderbooks": "lob", "raw": "lob",
               "order_flow_imbalance": "ofi", "orderflowimbalance": "ofi"}
    v = aliases.get(v, v)
    if v not in DATA_REPRESENTATIONS:
        raise ValueError(
            f"Unknown data representation {value!r}. Choose one of: "
            f"{', '.join(DATA_REPRESENTATIONS)}."
        )
    return v


def prompt_for_data_representation(default: str = "lob") -> str:
    """Interactively ask whether to run with raw LOB data or with OFI features added.

    Returns a canonical representation name. Falls back to `default` on a bare Enter or
    when stdin is not interactive (piped/batch runs).
    """
    import sys

    print("\nWhich type of data do you want to run with?")
    for i, r in enumerate(DATA_REPRESENTATIONS, start=1):
        marker = " (default)" if r == default else ""
        print(f"  [{i}] {r}{marker}\n      {DATA_REPRESENTATION_DESCRIPTIONS[r]}")

    if not sys.stdin or not sys.stdin.isatty():
        print(f"(non-interactive stdin) -> using default: {default}\n")
        return default

    while True:
        choice = input(
            f"\nEnter 1-{len(DATA_REPRESENTATIONS)} or a name [default: {default}]: "
        ).strip().lower()
        if choice == "":
            return default
        if choice.isdigit() and 1 <= int(choice) <= len(DATA_REPRESENTATIONS):
            return DATA_REPRESENTATIONS[int(choice) - 1]
        try:
            return normalize_representation_arg(choice)
        except ValueError as e:
            print(f"  {e}")


def resolve_representation(representation, default: str = "lob", interactive: bool = True) -> str:
    """Return a canonical representation name: use `representation` if given, else prompt."""
    representation = normalize_representation_arg(representation)
    if representation is not None:
        return representation
    if interactive:
        return prompt_for_data_representation(default=default)
    return default


def resolve_for_exchange(method, interactive: bool = True) -> str:
    """Top-level helper for the EXCHANGE path: ask data-type, then (if exchange) the
    normalization method. LOBSTER always returns 'rolling_5' without a method question.

    Returns the canonical normalization method to apply.
    """
    # If a method was supplied explicitly, honor it and skip the data-type question.
    method = normalize_method_arg(method)
    if method is not None:
        return method
    if not interactive:
        return "global"

    data_type = prompt_for_data_type(default="exchange")
    if data_type == "lobster":
        print("LOBSTER selected -> using the original 5-day rolling z-score.\n")
        return "rolling_5"
    return prompt_for_method(default="global")
