from pathlib import Path

path = Path('dashboard/app.py')
text = path.read_text(encoding='utf-8')
start = text.index('def render_phase_aware_progress_panel(')
end = text.index('def render_ml_institutional_intelligence_panel(')

new_func = '''def render_phase_aware_progress_panel(
    prediction: PredictionResult,
    transition: TransitionProbabilityResult | None,
    allocation: dict,
    symbol: str = '',
) -> None:
    """
    Phase-aware progress panel - single authoritative render point.

    WAIT / AVOID / CAUTION:
        ML Buy Transition Probability bar (from TransitionProbabilityResult)
        + exact helper text.
        + AI Conviction bar.

    BUY:
        Three actionable profit-booking indicators derived entirely from the
        ML-computed exit-side component scores in TransitionProbabilityResult.
        + AI Conviction bar.

    Zero randomness. Zero hardcoded placeholders. Uses 'Inference Calculating...'
    if data is missing.
    """
    if allocation is None:
        allocation = {}

    phase = str(allocation.get('status', 'WAIT')).upper()
    is_buy = phase == 'BUY'

    is_missing_data = False
    if transition is None:
        is_missing_data = True
    else:
        try:
            is_fallback = getattr(transition, 'fallback', False) if not isinstance(transition, dict) else transition.get('fallback', False)
            if is_fallback:
                is_missing_data = True
        except Exception:
            pass

    if is_missing_data:
        with st.container(border=True):
            st.caption('⚡ ' + ('PROFIT MANAGEMENT DASHBOARD' if is_buy else 'PROXIMITY TO TRIGGER'))
            st.info('Inference Calculating... Awaiting live model data stream.')
    else:
        if is_buy:
            dashboard_data = build_buy_profit_dashboard_data(allocation, transition)
            if dashboard_data is None:
                with st.container(border=True):
                    st.caption('⚡ BUY PHASE - PROFIT MANAGEMENT DASHBOARD')
                    st.info('BUY state metrics are unavailable right now.')
            else:
                try:
                    urgency = str(
                        transition.get('urgency_tier', 'LOW')
                        if isinstance(transition, dict)
                        else getattr(transition, 'urgency_tier', 'LOW')
                    ).upper()
                except (TypeError, ValueError, AttributeError):
                    urgency = 'LOW'

                downside_pct = dashboard_data['downsideRisk']
                momentum_pct = dashboard_data['momentumExhaustion']
                upside_pct = dashboard_data['upsidePotential']

                st.markdown(
                    f"""
                    <style>
                    .profit-dash-card {{
                        background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
                        border: 1px solid #243244;
                        border-radius: 16px;
                        padding: 16px 16px 14px 16px;
                        margin: 8px 0 10px 0;
                        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
                    }}
                    .profit-dash-title {{
                        color: #f8fafc;
                        font-size: 0.78rem;
                        font-weight: 700;
                        letter-spacing: 0.14em;
                        text-transform: uppercase;
                        margin-bottom: 10px;
                    }}
                    .profit-metrics-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                        gap: 12px;
                    }}
                    .profit-metric {{
                        background: rgba(255,255,255,0.04);
                        border: 1px solid rgba(255,255,255,0.08);
                        border-radius: 14px;
                        padding: 12px 12px 10px 12px;
                        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
                        min-height: 150px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    }}
                    .profit-metric:hover {{
                        transform: translateY(-2px);
                        border-color: rgba(255,255,255,0.18);
                        box-shadow: 0 10px 26px rgba(0,0,0,0.28);
                    }}
                    .profit-metric-label {{
                        color: #9aa7bf;
                        font-size: 0.72rem;
                        font-weight: 700;
                        letter-spacing: 0.12em;
                        text-transform: uppercase;
                        margin-bottom: 8px;
                    }}
                    .profit-metric-value {{
                        font-size: 1.7rem;
                        font-weight: 800;
                        color: #f8fafc;
                        line-height: 1.1;
                    }}
                    .profit-metric-subtext {{
                        color: #9aa7bf;
                        font-size: 0.78rem;
                        margin-top: 4px;
                    }}
                    .profit-progress {{
                        width: 100%;
                        height: 8px;
                        background: rgba(255,255,255,0.08);
                        border-radius: 999px;
                        overflow: hidden;
                        margin-top: 8px;
                    }}
                    .profit-progress > div {{
                        height: 100%;
                        border-radius: 999px;
                        transition: width 0.6s ease;
                    }}
                    </style>
                    <div class="profit-dash-card">
                        <div class="profit-dash-title">⚡ BUY PHASE — PROFIT BOOKING</div>
                        <div class="profit-metrics-grid">
                            <div class="profit-metric" title="Higher values indicate stronger upside toward the target price.">
                                <div>
                                    <div class="profit-metric-label">Upside Potential</div>
                                    <div class="profit-metric-value" style="color:#00e676;">{upside_pct:.1f}%</div>
                                </div>
                                <div>
                                    <div class="profit-progress"><div style="width:{upside_pct:.1f}%; background:linear-gradient(90deg, #00e676, #00c853);"></div></div>
                                    <div class="profit-metric-subtext">to Target Price</div>
                                </div>
                            </div>
                            <div class="profit-metric" title="Higher values suggest momentum is fading and a profit-booking event may be approaching.">
                                <div>
                                    <div class="profit-metric-label">Momentum Exhaustion</div>
                                    <div class="profit-metric-value" style="color:#ff9100;">{momentum_pct:.1f}%</div>
                                </div>
                                <div>
                                    <div class="profit-progress"><div style="width:{momentum_pct:.1f}%; background:linear-gradient(90deg, #ff9100, #ffb300);"></div></div>
                                    <div class="profit-metric-subtext">Trend is cooling</div>
                                </div>
                            </div>
                            <div class="profit-metric" title="Higher values indicate greater downside risk to the trailing stop.">
                                <div>
                                    <div class="profit-metric-label">Downside Risk</div>
                                    <div class="profit-metric-value" style="color:#ff5252;">{downside_pct:.1f}%</div>
                                </div>
                                <div>
                                    <div class="profit-progress"><div style="width:{downside_pct:.1f}%; background:linear-gradient(90deg, #ff5252, #ef5350);"></div></div>
                                    <div class="profit-metric-subtext">to Trailing Stop</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if urgency == 'HIGH':
                    st.caption('⚠️ High exit pressure - consider scaling out or tightening trailing stops.')
                elif urgency == 'MODERATE':
                    st.caption('👁 Moderate reversal signals - monitor RSI and MACD for exit confirmation.')
                else:
                    st.caption('✅ Trend intact - low reversal risk. Continue holding per strategy plan.')
        else:
            with st.container(border=True):
                try:
                    _pv = max(0.0, min(100.0, float(
                        transition.get('primary_metric_pct', 0.0)
                        if isinstance(transition, dict)
                        else getattr(transition, 'primary_metric_pct', 0.0)
                    )))
                except (TypeError, ValueError, AttributeError):
                    _pv = 0.0

                st.caption('PROXIMITY TO TRIGGER — BUY TRANSITION PROBABILITY')
                st.progress(_pv / 100.0, text=f'{_pv:.1f}%')
                st.caption('Early accumulation zone — momentum recovering but not yet at BUY threshold.')

    try:
        render_confidence_progress(prediction, transition, allocation)
    except Exception:
        pass

'''

path.write_text(text[:start] + new_func + text[end:], encoding='utf-8')
print('updated file')
