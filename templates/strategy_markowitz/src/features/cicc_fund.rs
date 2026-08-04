//! The CICC fundamental factor handbook (中金《量化基本面因子手册》), the
//! implementable subset of its catalog (图表169/170), hand-wired like
//! [`basic`](super::basic) — report-cadence data cannot ride the expression
//! engine's daily clock.
//!
//! # Conventions
//!
//! Flows (income statement, cash-flow statement) are annualized from their
//! YTD reports on the report effective dates ([`annualized`]) and turned into
//! TTM figures by [`ttm`]; balance-sheet stocks are used at their latest
//! carried value. The panels store assets and incomes debit-positive,
//! liabilities, equity, expenses and cash outflows credit-negative (the
//! crawler negates every 减-item so each statement's tree sums to zero);
//! formulas negate them where a positive magnitude is wanted. Where the
//! handbook says 平均 (average) balances (`AT`, `INVT`, `RAT`), the average
//! is `(current + year-ago) / 2` over a 365-day lag; the 变动 (delta, `*D`)
//! and 同比 (YoY growth, `*_YOY`) factors likewise compare against the level
//! 365 days ago, sampled daily by the self-recording [`rolling`] operators
//! (so a stock without a full prior year is `NaN` — the 次新 idiom).
//!
//! `DP` (股息率) sums the per-share cash dividends (CNY per share, ex-date
//! aligned) over the trailing 365 days — an empty window is a true `0`, no
//! payout, rather than missing — and divides by the raw close, so
//! `分红总额 / 总市值` cancels the share count.
//!
//! Every factor is a report-derived ratio whose cross-section is heavy-tailed
//! (denominators pass through zero), so the rank transform is part of each
//! definition: the catalog returns cross-sectional `[0, 1]` percentiles, for
//! which the [`winsorize_impute`](super::winsorize_impute) finalizer is
//! benign.
//!
//! # Deviations from the handbook, beyond spelling
//!
//! * **净利润 is the total** (`income_statement.profit`, minority interests
//!   included): the 归母净利润 column (`income_statement.parent_interests`)
//!   is a best-effort 减-item the source often omits, while the total is
//!   balanced by the statement tree. 净资产 *is* 归母 (the
//!   [`parent_equity`] approximation), matching the handbook's `DTE`.
//! * **营业收入 is read as 营业总收入** (`…profit.operating.income`) in the
//!   revenue ratios (`AT`, `NPM_TTM`, `OPM_TTM`, `SP_TTM`, `OR_YOY`, `RAT`):
//!   financial firms report no 营业收入 line — the crawler maps their whole
//!   income onto the total — so the total is the denominator that is reliable
//!   for every firm. 毛利 stays `营业收入 − 营业成本` (`GPMD`, `OPtoGR_TTM`),
//!   an industrial concept that correctly degenerates to `NaN` (imputed
//!   neutral) for financials.
//! * **ROIC's numerator is 营业利润 TTM** (operating profit), a constant-tax
//!   proxy for the handbook's 息前税后经营利润 (NOPAT); invested capital is
//!   `总资产 − 流动负债`.
//! * **`CSR`'s numerator is 货币资金** (monetary funds), the balance-sheet
//!   stand-in for the handbook's 现金及现金等价物.
//! * **`QR` (速动比率) omits 待摊费用** (deferred expenses), which has no
//!   current-asset column of its own: `(流动资产 − 存货 − 1年内到期的非流动
//!   资产 − 预付款) / 流动负债`.
//! * **`PEG_TTM` becomes `PEG_INV_TTM`**, its inverse
//!   (净利润增长率 × 100 × E/P): PE's sign flips with the earnings, so PEG is
//!   not monotone in anything — a negative PEG is closer in meaning to a
//!   large positive one than to a small one. The inverse is monotone in both
//!   components, the same reason the handbook itself quotes valuations as
//!   市盈率倒数/市净率倒数/市销率倒数 (`EP`/`BP`/`SP`) rather than PE/PB/PS.
//! * **上期 in the 变动 factors is read as 去年同期** (365 days ago), matching
//!   the 同比 factors; the handbook does not spell the lag out.
//! * **TTM is a 365-day daily-sampled mean** of the carried annualized flow
//!   (the [`basic`](super::basic) idiom): day-weighted across the ~4 reports
//!   in the window rather than report-weighted.
//!
//! # Omitted handbook factors
//!
//! * **Data the panels do not carry**: the 分析师 (analyst consensus), 公司
//!   治理 (governance, except `DPR_TTM`), 股东 (shareholder/institution
//!   counts) categories, and `EV2EBITDA` (needs depreciation & amortization,
//!   which lives in the low-availability indirect statements).
//! * **Single-quarter growth** (`*_Q_YOY`, `*_QOQ`, `NP_SD`, `OP_SD`): needs
//!   single-quarter flows, i.e. YTD differencing keyed by report period
//!   rather than by time.
//! * **Model-based factors**: `LPNP`, `NP_SUE0/1`, `OCFA` (rolling
//!   regressions), `QPT` and the composite factors (`Profit`, `Growth`,
//!   `Opt`, `Safe`, `Acc`, `QQC`, `Comp_opt`).
//! * **Time-series z-scores** (`*_Z`, e.g. `NP_Z`, `TOE_Z`): the handbook
//!   does not state their window.
//! * **`NP_Deducted_YOY`** (扣非): non-recurring items are not separable from
//!   the income panel.
//! * **`MC` and `FC`**: dropped as exact rank-duplicates of `Ln_MC` / `Ln_FC`
//!   under the cross-sectional rank transform.

use tradingflow::{
    data::Instant,
    graph::Builder,
    operators::{array, elem, rolling, signal, stats},
    ports::ArrayPortHandle,
    time::UnixTime,
};

use super::basic::{YEAR, annualized, change, growth, parent_equity, ttm};
use crate::data::MarketData;

/// Builds the CICC fundamental catalog, returning one `(name, ranked value)`
/// entry per factor, added in the handbook's category order. Every entry is
/// already percentile-ranked to `[0, 1]` (see the [module docs](self)).
pub fn build_features_cicc_fund(
    b: &mut Builder<Instant, UnixTime>,
    m: &MarketData,
) -> Vec<(String, ArrayPortHandle<f64, 1>)> {
    let daily = m.daily;
    let mut entries = Vec::new();
    let mut add = |name: &str, h: ArrayPortHandle<f64, 1>| {
        entries.push((name.to_string(), h));
    };
    // Shorthand for the pervasive quotient wiring.
    macro_rules! div {
        ($x:expr, $y:expr) => {
            b.op(elem::div(), ($x, $y))
        };
    }

    // ---- Shared building blocks ----
    let mc = b.op(elem::mul(), (m.close, m.field("shares.total")));
    let fc = b.op(elem::mul(), (m.close, m.field("shares.circulating")));
    let equity = parent_equity(b, m); // 归母净资产, positive
    let assets = m.field("balance_sheet.assets"); // 总资产, positive
    let debt = b.op(elem::neg(), m.field("balance_sheet.liab")); // 总负债
    let cur_assets = m.field("balance_sheet.assets.current"); // 流动资产
    let cur_liab = b.op(elem::neg(), m.field("balance_sheet.liab.current")); // 流动负债
    let cash = m.field("balance_sheet.assets.current.cash"); // 货币资金
    let inventories = m.field("balance_sheet.assets.current.inventories"); // 存货
    let receivables = m.field("balance_sheet.assets.current.receivables.notes_and_accounts");
    let taxes_payable = b.op(
        elem::neg(),
        m.field("balance_sheet.liab.current.payables.taxes"),
    ); // 应交税费, positive

    // Annualized-then-TTM flows. Income items gate on the income report
    // signals, cash-flow items on the cash-flow report signals.
    let inc = |b: &mut Builder<Instant, UnixTime>, name: &str| {
        let ann = annualized(b, m, m.income_report_signals, name);
        ttm(b, daily, ann)
    };
    let cf = |b: &mut Builder<Instant, UnixTime>, name: &str| {
        let ann = annualized(b, m, m.cf_report_signals, name);
        ttm(b, daily, ann)
    };
    let np_ttm = inc(b, "income_statement.profit"); // 净利润 TTM
    let op_ttm = inc(b, "income_statement.profit.operating"); // 营业利润 TTM
    let income_ttm = inc(b, "income_statement.profit.operating.income"); // 营业总收入 TTM
    let rev_ttm = inc(b, "income_statement.profit.operating.income.revenue"); // 营业收入 TTM
    let cost_ttm = inc(b, "income_statement.profit.operating.expenses.costs"); // 营业成本 (negative)
    let ocf_ttm = cf(b, "cash_flow_statement.change.operating"); // 经营现金流净额 TTM
    let ncf_ttm = cf(b, "cash_flow_statement.change"); // 净现金流 TTM
    let capex_ttm = cf(b, "cash_flow_statement.change.investing.out.assets"); // 购建资产 (negative)
    let tax_paid_ttm = cf(b, "cash_flow_statement.change.operating.out.taxes"); // 缴纳税费 (negative)

    let gross_ttm = b.op(elem::add(), (rev_ttm, cost_ttm)); // 毛利润 = 收入 − 成本
    let cogs_ttm = b.op(elem::neg(), cost_ttm); // 营业成本, positive
    let accruals_ttm = b.op(elem::sub(), (np_ttm, ocf_ttm)); // 应计利润
    let invested = b.op(elem::sub(), (assets, cur_liab)); // 投入资本 ≈ 总资产 − 流动负债
    let eps = div!(np_ttm, m.field("shares.total")); // 每股收益 TTM
    let fcf_ttm = b.op(elem::add(), (ocf_ttm, capex_ttm)); // 自由现金流 = 经营现金流 − 资本开支

    // 平均 (year-average) balances for the turnover ratios:
    // `(current + year-ago) / 2`.
    let avg = |b: &mut Builder<Instant, UnixTime>, h| {
        let lag = b.op(rolling::lag(YEAR), (daily, h));
        let two = b.op(elem::add(), (h, lag));
        b.op(array::map(|&x: &f64| x * 0.5), two)
    };
    let avg_assets = avg(b, assets);
    let avg_inventories = avg(b, inventories);
    let avg_receivables = avg(b, receivables);

    // Trailing-year per-share cash dividend: the dividend events collected
    // onto the daily clock (`NaN` off the ex-dates), summed over 365 days
    // with `min_count = 0` so an empty window is a true zero payout.
    let div_events = b.op(
        signal::collect(),
        (m.div_signals, m.field("dividends.cash"), daily),
    );
    let dps_ttm = b.op(rolling::sum(YEAR, 0), (daily, div_events));

    // ---- 盈利能力 (profitability) ----
    let roe = div!(np_ttm, equity); // 净利润 TTM / 净资产
    let roa = div!(np_ttm, assets); // 净利润 TTM / 总资产
    let cfoa = div!(ocf_ttm, assets); // 经营现金流净额 TTM / 总资产
    let roic = div!(op_ttm, invested); // 资本回报率 (see module docs)
    add("ROE_TTM", roe);
    add("ROED", change(b, daily, roe));
    add("ROA_TTM", roa);
    add("ROAD", change(b, daily, roa));
    add("CFOA", cfoa);
    add("CFOAD", change(b, daily, cfoa));
    add("ROIC_TTM", roic);
    add("ROICD", change(b, daily, roic));
    // TOE 应交税费占比 = (应交税费 − 上年同期应交税费 + 缴纳税费现金TTM) / 净资产.
    let taxes_delta = change(b, daily, taxes_payable);
    let tax_paid_pos = b.op(elem::neg(), tax_paid_ttm);
    let toe_num = b.op(elem::add(), (taxes_delta, tax_paid_pos));
    add("TOE", div!(toe_num, equity));

    // ---- 成长 (growth, TTM year-over-year) ----
    add("EPS_YOY", growth(b, daily, eps));
    add("NP_YOY", growth(b, daily, np_ttm));
    add("OCF_YOY", growth(b, daily, ocf_ttm));
    add("OP_YOY", growth(b, daily, op_ttm));
    add("OR_YOY", growth(b, daily, income_ttm));
    add("ROE_YOY", growth(b, daily, roe));
    add("TA_YOY", growth(b, daily, assets));

    // ---- 营运效率 (operating efficiency) ----
    let at = div!(income_ttm, avg_assets); // 资产周转率 = 收入 TTM / 平均总资产
    let gpm = div!(gross_ttm, rev_ttm); // 毛利率
    let invt = div!(cogs_ttm, avg_inventories); // 存货周转率 = 成本 TTM / 存货平均余额
    let opm = div!(op_ttm, income_ttm); // 营业利润率
    let rat = div!(income_ttm, avg_receivables); // 应收周转率 = 收入 TTM / 平均应收款
    add("AT", at);
    add("ATD", change(b, daily, at));
    add("GPMD", change(b, daily, gpm)); // 毛利率变动 (the level is not in the catalog)
    add("INVT", invt);
    add("INVTD", change(b, daily, invt));
    add("NPM_TTM", div!(np_ttm, income_ttm)); // 净利率
    add("OPM_TTM", opm);
    add("OPMD", change(b, daily, opm));
    add("OPtoGR_TTM", div!(op_ttm, gross_ttm)); // 营业利润 / 毛利润
    add("RAT", rat);
    add("RATD", change(b, daily, rat));

    // ---- 盈余质量 (earnings quality) ----
    let apr = div!(accruals_ttm, op_ttm); // 应计利润占比 = 应计利润 TTM / 营业利润 TTM
    let csr = div!(cash, cur_liab); // 现金比率
    add("APR_TTM", apr);
    add("APRD", change(b, daily, apr));
    add("CSR", csr);
    add("CSRD", change(b, daily, csr));

    // ---- 安全性 (safety) ----
    let ccr = div!(ocf_ttm, cur_liab); // 现金流动负债比率
    let cur = div!(cur_assets, cur_liab); // 流动比率
    let debt_asset = div!(debt, assets); // 资产负债比
    let dte = div!(debt, equity); // 产权比率
    // 速动比率 (quick assets / current liabilities; see module docs).
    let q1 = b.op(elem::sub(), (cur_assets, inventories));
    let q2 = b.op(
        elem::sub(),
        (q1, m.field("balance_sheet.assets.current.noncurrent_due")),
    );
    let quick = b.op(
        elem::sub(),
        (q2, m.field("balance_sheet.assets.current.prepayments")),
    );
    let qr = div!(quick, cur_liab);
    add("CCR", ccr);
    add("CCRD", change(b, daily, ccr));
    add("CUR", cur);
    add("CURD", change(b, daily, cur));
    add("Debt_Asset", debt_asset);
    add("DAD", change(b, daily, debt_asset));
    add("DTE", dte);
    add("DTED", change(b, daily, dte));
    add("QR", qr);
    add("QRD", change(b, daily, qr));

    // ---- 估值 (valuation) & 分红 (payout) ----
    add("BP_LR", div!(equity, mc)); // 净资产 / 总市值
    add("DP", div!(dps_ttm, m.close)); // 股息率 = 分红 TTM / 总市值 (per share)
    let div_total = b.op(elem::mul(), (dps_ttm, m.field("shares.total")));
    add("DPR_TTM", div!(div_total, np_ttm)); // 股利支付率 = 现金分红 / 净利润
    add("EP_TTM", div!(np_ttm, mc)); // 净利润 TTM / 总市值
    add("FCFP_TTM", div!(fcf_ttm, mc)); // 自由现金流 TTM / 总市值
    add("NCFP_TTM", div!(ncf_ttm, mc)); // 净现金流 TTM / 总市值
    add("OCFP_TTM", div!(ocf_ttm, mc)); // 经营现金流 TTM / 总市值
    // The handbook's PEG (= PE TTM / (净利润增长率 × 100)) is inverted, like
    // its own 市盈率倒数: PE's sign flips with the earnings, so a negative
    // PEG reads as "cheap" when it means "shrinking or loss-making". The
    // inverse 增长率 × 100 × E/P is monotone in both components.
    let np_yoy = growth(b, daily, np_ttm);
    let np_yoy_pct = b.op(array::map(|&x: &f64| x * 100.0), np_yoy);
    let ep = div!(np_ttm, mc);
    add("PEG_INV_TTM", b.op(elem::mul(), (np_yoy_pct, ep)));
    add("SP_TTM", div!(income_ttm, mc)); // 营业收入 TTM / 总市值

    // ---- 规模 (size) ----
    add("FC_MC", div!(fc, mc)); // 流通市值 / 总市值
    add("Ln_FC", b.op(elem::ln(), fc)); // 流通市值对数
    add("Ln_MC", b.op(elem::ln(), mc)); // 总市值对数

    // The rank transform is part of every factor's definition (see the
    // module docs): percentile to `[0, 1]`, a missing value staying `NaN` for
    // the finalizer's mean imputation (the neutral median `0.5`).
    entries
        .into_iter()
        .map(|(name, h)| (name, b.op(stats::percentile(), h)))
        .collect()
}
