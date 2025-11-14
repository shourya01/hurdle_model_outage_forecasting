# Feature Vector Definition

For each county \(i\) and timestep \(t\), both outage pipelines build a dense covariate vector \(x_{i,t}\) that summarises recent outage/weather history plus seasonal signals:
\[
o_{i,t} = f(x_{i,t}; \theta).
\]

## Structure of \(x_{i,t}\)

\[
x_{i,t} = \Big[
\mathrm{hist\_len}_{i,t},\;
\{o_{i,t-\ell}\}_{\ell=1}^{48},\;
\mathrm{runlen}_{i,t},\;
\{h^{(k)}_{i,t-1},\, h^{(k)}_{i,t-1}-h^{(k)}_{i,t-2}\}_{k \in \mathcal{H}},\;
\tau(t),\;
\text{county\_idx}(i)
\Big].
\]

| Component | Description |
| --- | --- |
| \(\mathrm{hist\_len}_{i,t}\) | Effective history depth \(\min(t, 48)\); indicates how many lags are populated. |
| \(\{o_{i,t-\ell}\}\) | Dense outage lags for \(\ell=1…48\); missing samples are treated as zero before inclusion. |
| Outage roll sums/maxima | Sliding-window sums and maxima over \(\{6,12,24\}\) hours to capture burstiness. |
| \(\mathrm{runlen}_{i,t}\) | Hours since the last nonzero outage; `NaN` when no history exists. |
| \(h^{(k)}_{i,t-1}\) | Latest value for each shared weather feature \(k \in \mathcal{H}\) (blh, cape, …, v10). |
| \(h^{(k)}_{i,t-1}-h^{(k)}_{i,t-2}\) | First-order difference of each weather feature; set to zero when only one past sample exists. |
| Weather roll stats | Rolling maxima and sums of each weather feature over \(\{6,12,24\}\) hours. |
| Interactions | Products \(o_{i,t-1} \times h^{(k)}_{i,t-1}\) linking the latest outage to each hazard. |
| \(\tau(t)\) | Seasonal encoding \([\sin(\frac{2\pi}{24}\text{HoD}), \cos(\frac{2\pi}{24}\text{HoD}), \sin(\frac{2\pi}{7}\text{DoW}), \cos(\frac{2\pi}{7}\text{DoW})]\). |
| \(\text{county\_idx}(i)\) | LightGBM categorical ID for county \(i\). |

Both the two-stage (binary + Tweedie) and n-stage (multiclass + Tweedie tail) heads consume this same \(x_{i,t}\); only the prediction layer differs.
