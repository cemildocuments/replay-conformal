# Data attribution

The following datasets are distributed by the UCI Machine Learning Repository under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Their original dataset pages identify the creators and licenses. Original archives are preserved in `raw/`; transformed numerical records are in `processed/`.

| Dataset | Creators | Year | DOI |
|---|---|---:|---|
| Abalone | Warwick Nash, Tracy Sellers, Simon Talbot, Andrew Cawthorn, Wes Ford | 1994 | [10.24432/C55C7W](https://doi.org/10.24432/C55C7W) |
| Airfoil Self-Noise | Thomas Brooks, D. Pope, Michael Marcolini | 1989 | [10.24432/C5VW2C](https://doi.org/10.24432/C5VW2C) |
| Concrete Compressive Strength | I-Cheng Yeh | 1998 | [10.24432/C5PK67](https://doi.org/10.24432/C5PK67) |
| Energy Efficiency | Athanasios Tsanas, Angeliki Xifara | 2012 | [10.24432/C51307](https://doi.org/10.24432/C51307) |
| Wine Quality | Paulo Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis | 2009 | [10.24432/C56S3T](https://doi.org/10.24432/C56S3T) |

## Transformations

Exact complete-row duplicates are removed before partitioning. The red and white wine datasets remain separate tasks. Abalone sex is one-hot encoded and ring count is the response. Energy uses the heating-load response and the eight input variables. The original numerical units are preserved. The CSV columns `x0`, `x1`, and so on follow the original input order; the abalone categorical indicator columns follow the seven numerical columns in F, I, M order. The last column is `target`.

Retained record counts and exact SHA-256 hashes are recorded in `dataset_summary.csv`. Deduplication defines a canonical benchmark distribution and is not a claim that identical physical measurements came from the same real-world entity.

Replay multiplicities are generated experimentally after splitting. The methods assign extra copies by low or high calibration score, balanced replication, or a random subset of available copy slots. Every source record is retained at least once. The replay scenarios are derived benchmark conditions, not newly collected observational data.

## License notice

The CC BY 4.0 license permits redistribution and adaptation with attribution. Retain these credits, DOI links, license link, and transformation notices when sharing the source or processed datasets. The original creators do not endorse this experimental study.
