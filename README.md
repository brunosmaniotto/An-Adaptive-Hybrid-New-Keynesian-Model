# An Adaptive-Hybrid New-Keynesian Model

**Bruno Cittolin Smaniotto** | UC Berkeley

## Overview

We develop an adaptive model of inflation expectations in which agents dynamically choose between anchoring to the central bank's target and extrapolating from recent inflation based on forecasting performance. When inflation is stable, anchored forecasts outperform and credibility remains high; when inflation deviates persistently, agents de-anchor and inflation becomes self-reinforcing. The framework provides a dynamic theory of central bank credibility in which credibility and inflation outcomes evolve together through mutual feedback. Simulations show the framework is consistent with key features of the Volcker disinflation, the missing disinflation of 2008--09, and the post-pandemic inflation surge.

## Quick Start

```bash
pip install -r requirements.txt
python code/run_all.py
```

See [REPLICATION.md](REPLICATION.md) for detailed instructions and expected output.

## Repository Structure

```
code/
  models/          Core model implementations (MAB learning, FIRE, full NK)
  simulations/     Simulation scripts, organized by paper section
  plotting/        Figure generation
  tables/          Table generation
  empirical/       Empirical analysis (rolling persistence, Kalman filter)
  run_all.py       Master replication script
data/
  raw/             U.S. CPI data from FRED
output/
  figures/         Generated figures (PNG and PDF)
  tables/          Generated tables (CSV)
  simulations/     Cached simulation data
  empirical/       Empirical analysis output
manuscript/
  paper.tex        Main manuscript
  references.bib   Bibliography
```

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

## Citation

```bibtex
@article{smaniotto2026adaptive,
  title={An Adaptive-Hybrid New-Keynesian Model},
  author={Smaniotto, Bruno Cittolin},
  year={2026},
  journal={Working Paper}
}
```

## License

MIT License -- see [LICENSE](LICENSE).
