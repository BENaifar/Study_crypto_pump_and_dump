# %% [markdown]
# # I.1 — Literature synthesis
#
# **Role E2 (bibliography and report).** The brief asks four questions, answered from [A1] (Hu et al., SIGMOD 2023)
# and [A2] (Pacheco et al., ICWSM 2021) and illustrated, where possible, with the data of this repository:
#
# 1. Why do organisers target mid-capitalisation coins?
# 2. What do intra-channel homogeneity and inter-channel heterogeneity of targets mean?
# 3. How does [A1] handle the cold start?
# 4. What do the European regulations MAR and MiCA provide on market manipulation?
#
# **The pump-and-dump scheme in brief** ([A1] §2, [4]). Organisers announce the exchange and the time in advance but
# keep the target secret. Insiders (organisers, VIP members who pay for early access) buy before the announcement (the
# *pre-pump*, which [A1] dates to up to about 60 hours before). At the announced minute the coin is revealed; members
# rush in, and the price can multiply within a minute (+250 % for NAS on Binance on 22 August 2021). Insiders then
# sell into the buying wave (the *dump*), and the price collapses within minutes, leaving late buyers with the loss.

# %%
import pandas as pd

from pnd_paper.data import pd_log, coingecko
from pnd_paper.report import Reporter, md, show

R = Reporter("I1_literature_synthesis")
log = pd_log.load_pd_log()
ev = pd_log.binance_btc_events(log)

# %% [markdown]
# ## 1. Why mid-capitalisation coins?
#
# A pump must satisfy two conflicting constraints.
#
# * **Movability.** The price impact of a buy order of size $Q$ grows with $Q$ relative to the coin's liquidity
#   (order-book depth, which scales with capitalisation and volume). A large-cap coin cannot be moved by a few
#   thousand members with modest budgets. [A1] notes that organisers choose coins whose market is thin enough to
#   be pushed.
# * **Exit liquidity.** Insiders profit only if they can *sell* their position at the top. For a micro-cap coin the
#   order book on the sell side is so shallow that insiders cannot unload without crashing the price themselves, and
#   very small coins are often not listed on the large exchanges where the members trade (Binance requires a listing).
#   Detection risk also grows: an obscure coin multiplying by ten is conspicuous.
#
# Hence the "sweet spot": coins small enough to be moved, large enough to be listed on a major exchange and to absorb
# the insiders' sales. [A1] §4.3 shows that pumped coins concentrate in a band of market capitalisation and audience
# (Alexa rank, Reddit subscribers). Our data (I.2) make it quantitative:

# %%
gk = coingecko.load_all()
t = gk[gk.source == "targets"].market_cap_usd.dropna()
c = gk[gk.source == "candidates"].market_cap_usd.dropna()
q = pd.DataFrame({"targets": t.quantile([0.1, 0.25, 0.5, 0.75, 0.9]) / 1e6,
                  "coins listed on Binance (same dates)": c.quantile([0.1, 0.25, 0.5, 0.75, 0.9]) / 1e6})
q.index = [f"{int(100 * i)}th percentile" for i in q.index]
R.table(q, "mcap_quantiles_musd")
show(q, "{:.1f}")
md(f"Market capitalisation in millions of USD, three days before the pump. The median target "
   f"(USD {t.median() / 1e6:.0f}M) is about {c.median() / t.median():.0f} times smaller than the median coin listed on Binance "
   f"at the same dates, and 90 % of targets lie between USD {t.quantile(0.05) / 1e6:.0f}M and "
   f"USD {t.quantile(0.95) / 1e6:.0f}M (5th–95th percentiles): small to mid caps relative to the exchange, almost never "
   "the large caps.")

# %% [markdown]
# ## 2. Intra-channel homogeneity and inter-channel heterogeneity
#
# * **Intra-channel homogeneity** ([A1] §4.3): the coins pumped by *one* channel resemble each other in market
#   capitalisation, popularity and age. A channel has a stable community with a given total budget and risk
#   appetite; the organisers learn which coins this community can move and keep choosing similar ones. [A1] also
#   finds that word embeddings of coins pumped by the same channel are more similar (cosine 0.92) than those of
#   random pumped coins (0.80) or random coins (0.72).
# * **Inter-channel heterogeneity**: different channels work in different niches; a large channel with tens of
#   thousands of members targets bigger coins than a small one.
#
# **Consequence for prediction.** A single global "pumpability" score misses this structure: the relevant question is
# "which coin would *this* channel pick next?". [A1] therefore conditions the prediction on the channel's sequence of
# past targets (the SNN). It also explains the high re-pump rate (60.1 % of pumped coins are pumped again, §4.1).
# In this project, I.4 tests homogeneity with a permutation test (rejecting the null of exchangeable channel labels),
# and II.B/II.C measure how much channel-history features improve ranking.
#
# [A2] provides the complementary *network* view used in I.5: accounts (here channels) that share rare behavioural
# traces (here the same coin in the same hour) are linked, and dense groups reveal coordination that no single
# account's behaviour shows.

# %% [markdown]
# ## 3. Cold start in [A1]
#
# The SNN represents every coin by a learned embedding $E(c)$, shared by the target candidate and the history
# elements. Two cold-start situations break it ([A1] §5.3): (i) the coin pumped in the test period has never been
# pumped during training, so its embedding received no signal from positive examples; (ii) the coin did not exist
# during training at all. [A1] replaces $E(c)$ by the **word embedding of the coin symbol**, pre-trained with SkipGram
# (and CBOW) on a large corpus of Telegram messages from crypto channels and groups. Coins discussed in similar
# contexts get similar vectors even if they were never pumped. The variant, called SNN$_S$, improves HR@1 from 0.260
# to 0.277 and HR@30 from 0.797 to 0.823 (Table 6). How frequent is the problem in our data?

# %%
train_coins = set(ev[ev.split == "train"].coin)
seen = ev[ev.split == "test"].coin.isin(train_coins)
ever = ev.sort_values("timestamp").assign(prev=lambda d: d.groupby("coin").cumcount() > 0)
cs = pd.DataFrame({"value": [len(ev[ev.split == "test"]), (~seen).mean(), (~ever[ever.split == "test"].prev).mean()]},
                  index=["test events (Binance/BTC)", "share of test targets never pumped in the training period",
                         "share of test targets never pumped before (any earlier event)"])
R.table(cs, "cold_start_share")
show(cs, "{:.3f}")
md(f"About **{(~seen).mean():.0%}** of test targets were never pumped during training: for them the learned coin "
   "embedding carries no pump-specific signal, which is why a representation learned from text (or from the coin's "
   "market features) is needed. II.C rebuilds SNN$_S$ with embeddings learned from the authors' Telegram corpus.")

# %% [markdown]
# ## 4. European regulation: MAR and MiCA
#
# **MAR — Regulation (EU) No 596/2014 on market abuse.**
# * *Scope* (Art. 2): financial instruments admitted to trading on a regulated market, an MTF or an OTF. Most crypto-assets
#   (bitcoin, utility tokens) are **not** financial instruments under MiFID II, so MAR generally does not reach the
#   pumps studied here, a gap that motivated MiCA.
# * *Definition* (Art. 12): market manipulation covers transactions or orders that give false or misleading signals
#   about supply, demand or price, or secure the price at an abnormal level (12(1)(a)); transactions using fictitious
#   devices or deception (12(1)(b)); and the dissemination of false or misleading information, including rumours, via
#   the media or the internet (12(1)(c)). Art. 12(2) gives examples, among them securing a dominant position and taking
#   positions and then voicing opinions about the instrument to profit from the price impact.
# * *Prohibition* (Art. 15) of manipulation and attempted manipulation; *prevention and detection* (Art. 16): market
#   operators and persons professionally arranging or executing transactions must maintain effective arrangements and
#   report suspicious orders and transactions (STORs) to the competent authority; *sanctions* (Art. 30): administrative
#   fines of at least EUR 5 million for individuals and EUR 15 million or 15 % of turnover for legal persons for
#   manipulation, alongside criminal sanctions under Directive 2014/57/EU (CSMAD).
# * Commission Delegated Regulation (EU) 2016/522, Annex II, lists example practices for the indicators of
#   manipulation of MAR Annex I; **"pump and dump"** is named explicitly (Section 1, point 4(c)): taking a long
#   position, disseminating misleading positive information to raise the price, and selling at the inflated price.
#
# **MiCA — Regulation (EU) 2023/1114 on markets in crypto-assets.** Title VI (Art. 86–92) transposes the MAR regime to
# crypto-assets admitted (or for which admission is requested) to trading on a crypto-asset trading platform:
# * Art. 87–88: inside information and its public disclosure by issuers; Art. 89–90: prohibition of insider dealing and
#   unlawful disclosure. The insiders' pre-pump buying (I.3) is exactly the use of inside information about the
#   upcoming target.
# * **Art. 91: prohibition of market manipulation.** It reproduces MAR's catalogue and adds a crypto-specific
#   example: taking advantage of occasional or regular access to traditional or electronic media, including social
#   media, by voicing an opinion about a crypto-asset after having taken positions in it, without disclosing the
#   conflict of interest. Telegram pump announcements fall squarely within this.
# * **Art. 92: prevention and detection.** Any person professionally arranging or executing transactions in
#   crypto-assets (in particular trading platforms such as Binance's EU entities) must have effective arrangements,
#   systems and procedures to prevent and detect market abuse, and must report reasonable suspicions to the competent
#   authority without delay (STOR). ESMA's RTS on these arrangements and the report template were adopted as Delegated
#   Regulation (EU) 2025/885.
# * *Application*: the market-abuse title applies from **30 December 2024**; national competent authorities enforce it
#   with administrative sanctions (Art. 111).
#
# **Link with this project.** The detectors of II.A (alarms before $\tau$) and the target ranking of II.B/II.C are
# examples of the "arrangements, systems and procedures" Art. 92 MiCA requires from platforms. Their false-positive
# rate matters: each alarm triggers a review, and an unfounded STOR has a cost for the platform and for the investors
# whose orders are delayed.

# %% [markdown]
# ## References
#
# 1. S. Hu, Z. Zhang, S. Lu, B. He, Z. Li. *Sequence-based target coin prediction for cryptocurrency pump-and-dump.*
#    Proc. ACM Manag. Data 1(1), art. 6, 2023 (arXiv:2204.12929).
# 2. D. Pacheco, P.-M. Hui, C. Torres-Lugo, B. T. Truong, A. Flammini, F. Menczer. *Uncovering coordinated networks on
#    social media: methods and case studies.* ICWSM 15:455–466, 2021.
# 3. S. Hu et al. *Pump-and-Dump-Detection-on-Cryptocurrency*, GitHub repository.
# 4. J. Xu, B. Livshits. *The anatomy of a cryptocurrency pump-and-dump scheme.* USENIX Security, 2019.
# 5. Regulation (EU) No 596/2014 (MAR), OJ L 173, 12.6.2014; Commission Delegated Regulation (EU) 2016/522, Annex II.
# 6. Regulation (EU) 2023/1114 (MiCA), OJ L 150, 9.6.2023, Title VI, Art. 86–92; ESMA Interactive Single Rulebook,
#    MiCA Art. 91 and 92: https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/mica/article-91-prohibition-market-manipulation
# 7. ESMA, *Final Report — Draft RTS on arrangements, systems and procedures to detect and report suspected market abuse
#    under MiCA (Art. 92(2))*, ESMA75-453128700-1278, December 2024.
# 8. O. Levy, Y. Goldberg. *Neural word embedding as implicit matrix factorization.* NeurIPS, 2014.
