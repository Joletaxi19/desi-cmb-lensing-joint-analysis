```mermaid
graph TD
  subgraph observations
    A1["LRG maps (2048)"] -->|mask & full_master| B1["lrg_galaxy_auto.json"]
    A2["PR4 kappa map (2048)"] -->|mask & full_master| B2["pr4_kappa_auto.json"]
  end

  subgraph theory
    C0["cls_LRGxPR4_bestFit.txt (+ Nl_kk)"] -->|pixwin × filt × bin LEDGES| C1["theory_bandpowers.json"]
  end

  B1 --> D["plot_coeffs_auto.py"]
  B2 --> D
  C1 --> D
  D --> E["figure finale"]