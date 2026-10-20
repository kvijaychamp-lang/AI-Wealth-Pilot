# Live Streaming Row Fix - Code Locations

## Summary of Changes

**File**: `dashboard/app.py`
**Function**: `run_streaming_state_loop()` (Lines 1857-2003)

### Changes Made

#### 1. Market Closed Path (Lines 1874-1906)
**Old Behavior**:
```python
if not market_active:
    ...
    return df_display, chart_df, current_price_display, price_source
```
- Returned `df_display` WITHOUT appending live row
- Table showed only historical data

**New Behavior**:
```python
if not market_active:
    ...
    tick_time = datetime.now(tz=NSE_TIMEZONE)
    df_streaming = append_live_ltp_row(df_display, ltp=current_price_display, tick_time=tick_time)
    df_streaming = compute_display_indicators(df_streaming)
    return df_streaming, chart_df, current_price_display, price_source
```
- Appends live row with current price
- Table now shows historical + live tick

#### 2. Fetch Error with Cached Price (Lines 1945-1973)
**Old Behavior**:
```python
if _is_internet_available():
    if cached_live is not None:
        ...
        return df_display, chart_df, float(cached_live), 'Live LTP'
```
- Returned `df_display` WITHOUT appending live row

**New Behavior**:
```python
if _is_internet_available():
    if cached_live is not None:
        ...
        df_streaming = append_live_ltp_row(df_display, ltp=float(cached_live), tick_time=tick_time)
        df_streaming = compute_display_indicators(df_streaming)
        return df_streaming, chart_df, float(cached_live), 'Live LTP (Cached)'
```
- Appends live row with cached price
- Recomputes indicators

#### 3. Offline Simulation Path (Lines 1975-2003)
**Old Behavior**:
```python
# No internet
logger.warning(...)
...
return df_display, chart_df, float(ticking_price), 'Simulated (Offline)'
```
- Returned `df_display` WITHOUT appending live row

**New Behavior**:
```python
# No internet
logger.warning(...)
...
df_streaming = append_live_ltp_row(df_display, ltp=float(ticking_price), tick_time=tick_time)
df_streaming = compute_display_indicators(df_streaming)
return df_streaming, chart_df, float(ticking_price), 'Simulated (Offline)'
```
- Appends live row with simulated price
- Recomputes indicators

## Verification Checklist

### Code Locations to Verify

| Line Range | Description | Status |
|-----------|-------------|--------|
| 1857-1906 | Market closed: append live row | ✅ Updated |
| 1909-1938 | Market open success: append live row | ✅ Unchanged (working) |
| 1945-1973 | Fetch error cached: append live row | ✅ Updated |
| 1975-2003 | Offline simulation: append live row | ✅ Updated |

### Expected Result

**Before**: 
- Table shows 10 rows (2026-06-03 to 2026-06-12)
- Index 0-9 only
- No live row

**After**:
- Table shows 11 rows (2026-06-03 to 2026-06-12, + live row)
- Index 0-10
- Live row at index 10 with timestamp "2026-06-15 HH:MM:SS"
- Indicators calculated for live row

## Key Functions Used

- `append_live_ltp_row()` - Appends/upserts live tick to dataframe
- `compute_display_indicators()` - Recalculates technical indicators
- `datetime.now(tz=NSE_TIMEZONE)` - Gets current server time as tick timestamp

## Testing

The fix ensures the live row appears in ALL scenarios:

| Scenario | Before | After |
|----------|--------|-------|
| Market closed (weekend/holiday) | ❌ No live row | ✅ Live row with current price |
| Market open, fetch fails | ❌ No live row | ✅ Live row with cached price |
| Offline/no internet | ❌ No live row | ✅ Live row with simulated price |
| Market open, fetch succeeds | ✅ Live row | ✅ Live row (unchanged) |

## Integration Points

The fix integrates with existing bulletproof Date handling:

1. Live row is created via `append_live_ltp_row()` which:
   - Uses `_extract_and_validate_timestamp()` for safe timestamp extraction
   - Uses `_match_timezone_to_historical()` for timezone matching
   - Guarantees Date is never None or 'None'

2. Date validation after append:
   - Converts to datetime with coerce
   - Drops NaT rows (failsafe)
   - Falls back to display frame if all rows removed

3. Indicator recomputation:
   - Called AFTER Date validation
   - Ensures indicators calculated on data WITH live row

## No Breaking Changes

✅ All changes are backward compatible
✅ Existing working paths (market open success) unchanged
✅ Enhanced error paths with live row appending
✅ Date validation consistent with previous fix
